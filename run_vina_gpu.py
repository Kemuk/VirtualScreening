#!/usr/bin/env python3
"""
run_vina_gpu_tqdm.py

Docking with QuickVina2-GPU:
- Uses tqdm.contrib.concurrent.process_map for batch-level parallel jobs
- Writes batches to scratch space
- Single progress bar (batches)
"""

import argparse, os, subprocess, tempfile, shutil
from pathlib import Path
import pandas as pd
from tqdm.contrib.concurrent import process_map

ENV_SETUP = (
    "ulimit -s 8192 || true; "
    "module purge || true; "
    "module load Boost/1.77.0-GCC-11.2.0 CUDA/12.0.0 || true"
)

# --------------------- Helpers ---------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="LIT_PCBA/vina_boxes.csv")
    ap.add_argument("--vina_bin", default="vina-gpu-dev/QuickVina2-GPU-2-1")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--scratch", default="/tmp", help="Scratch directory for temporary batch folders")
    return ap.parse_args()

def detect_num_gpus():
    try:
        out = subprocess.check_output("nvidia-smi -L | wc -l", shell=True)
        return int(out.decode().strip())
    except Exception:
        return 1

def get_ligands(cfg: Path):
    lig_dir = None
    with open(cfg) as f:
        for line in f:
            if line.strip().startswith("ligand_directory"):
                lig_dir = Path(line.split("=")[1].strip())
                if not lig_dir.is_absolute():
                    lig_dir = Path(cfg).parent / lig_dir
    return list(lig_dir.glob("*.pdbqt")) if lig_dir and lig_dir.exists() else []

def chunked(seq, size):
    return (seq[i:i+size] for i in range(0, len(seq), size))

def iter_batches(df, batch_size):
    configs = (
        Path(row[c]) for _, row in df.iterrows()
        for c in ("config_actives", "config_inactives")
        if row.get(c) and Path(row[c]).exists()
    )

    return (
        (cfg, i, batch)
        for cfg in configs
        for ligands in [get_ligands(cfg)]
        for i, batch in enumerate(chunked(ligands, batch_size))
        if batch
    )

# --------------------- Worker ---------------------
def run_batch(args):
    cfg, batch_idx, batch_ligs, vina_bin, gpu_id, scratch = args
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"{cfg.stem}_batch{batch_idx}_", dir=scratch))
    lig_batch_dir = tmp_dir / "ligs"
    lig_batch_dir.mkdir(parents=True, exist_ok=True)
    for lig in batch_ligs:
        shutil.copy(lig, lig_batch_dir)

    batch_cfg = tmp_dir / f"config_batch_{batch_idx}.txt"
    with open(cfg) as f, open(batch_cfg, "w") as fout:
        for line in f:
            if line.strip().startswith("ligand_directory"):
                fout.write(f"ligand_directory = {lig_batch_dir}\n")
            elif line.strip().startswith("output_directory"):
                out_dir = Path(line.split("=")[1].strip())
                if not out_dir.is_absolute():
                    out_dir = cfg.parent / out_dir
                out_dir = out_dir / f"batch_{batch_idx}"
                out_dir.mkdir(parents=True, exist_ok=True)
                fout.write(f"output_directory = {out_dir}\n")
            else:
                fout.write(line)

    log_file = tmp_dir / f"docking_gpu{gpu_id}.log"
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} {ENV_SETUP}; {vina_bin} --config {batch_cfg}"
    subprocess.call(cmd, shell=True, stdout=open(log_file, "w"),
                    stderr=subprocess.STDOUT, executable="/bin/bash")

    return len(batch_ligs)

# --------------------- Main ---------------------
def main():
    args = parse_args()
    num_gpus = args.gpus or detect_num_gpus()
    vina_bin = Path(args.vina_bin).resolve()
    if not vina_bin.exists():
        raise FileNotFoundError(f"QuickVina2-GPU binary not found: {vina_bin}")

    df = pd.read_csv(args.csv)

    # Prepare all batches
    all_batches = list(iter_batches(df, args.batch_size))
    total_ligands = sum(len(batch) for _, _, batch in all_batches)
    total_batches = len(all_batches)

    # Assign batches round-robin to GPUs
    tasks = [
        (cfg, batch_idx, batch, str(vina_bin), i % num_gpus, Path(args.scratch))
        for i, (cfg, batch_idx, batch) in enumerate(all_batches)
    ]

    # Run with tqdm process_map (batch-level bar)
    results = process_map(run_batch, tasks, max_workers=num_gpus,
                          total=total_batches, unit="batch", desc="Docking",
                          chunksize=10)

    print(f"\nDocking finished: {sum(results)}/{total_ligands} ligands processed in {total_batches} batches")

if __name__ == "__main__":
    main()
