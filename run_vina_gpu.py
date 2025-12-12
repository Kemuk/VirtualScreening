#!/usr/bin/env python3
"""
Sequential (non-MPI) docking with QuickVina2-GPU.

- Reads vina_boxes.csv
- Expands each config into per-ligand jobs
- Runs ligands sequentially (on a single GPU)
- Outputs go directly into output folders
- One manifest.csv per output folder
- Supports:
    --dry_run N   : Run only the first N ligands total
    --test_run    : Run only the first batch (first ligand) from each config
"""

import argparse, os, subprocess, csv, time
from pathlib import Path
import pandas as pd
from tqdm import tqdm

ENV_SETUP = (
    "ulimit -s 8192 || true; "
    "module purge || true; "
    "module load Boost/1.77.0-GCC-11.2.0  Python/3.9.6-GCCcore-11.2.0 "
    "CUDA/12.0.0  OpenMPI/4.1.1-GCC-11.2.0 || true"
)

# --------------------- Helpers ---------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="LIT_PCBA/vina_boxes.csv",
                    help="vina_boxes.csv with config_actives,config_inactives")
    ap.add_argument("--vina_bin", default="vina-gpu-dev/QuickVina2-GPU-2-1")
    ap.add_argument("--threads", type=int, default=8000,
                    help="Threads per job (default: 8000)")
    ap.add_argument("--dry_run", type=int, default=0,
                    help="If >0, only run this many ligands for testing")
    ap.add_argument("--test_run", action="store_true",
                    help="If set, only run the first ligand from each config")
    ap.add_argument("--gpu_id", type=int, default=0,
                    help="Which GPU to use (default: 0)")
    return ap.parse_args()

# --------------------- Job expansion ---------------------
def expand_config(cfg_path, threads, test_run=False):
    """Yield jobs for each ligand in a config file."""
    cfg = Path(cfg_path)
    receptor, lig_dir, out_dir = None, None, None
    center, size = {}, {}

    with open(cfg) as f:
        for line in f:
            parts = line.strip().split("=")
            if len(parts) != 2:
                continue
            key, val = parts[0].strip(), parts[1].strip()
            if key == "receptor":
                receptor = Path(val)
                if not receptor.is_absolute():
                    receptor = cfg.parent / receptor
            elif key == "ligand_directory":
                lig_dir = Path(val)
                if not lig_dir.is_absolute():
                    lig_dir = cfg.parent / lig_dir
            elif key in ("out", "output_directory"):
                out_dir = Path(val)
                if not out_dir.is_absolute():
                    out_dir = cfg.parent / out_dir
            elif key.startswith("center_"):
                center[key] = float(val)
            elif key.startswith("size_"):
                size[key] = float(val)

    if not (receptor and lig_dir and out_dir):
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    ligands = sorted(lig_dir.glob("*.pdbqt"))

    if test_run:
        ligands = ligands[:1]  # only first ligand

    for i, lig in enumerate(ligands):
        output_file = out_dir / f"{lig.stem}_out.pdbqt"
        yield {
            "receptor": str(receptor.resolve()),
            "ligand": str(lig.resolve()),
            "output_file": str(output_file.resolve()),
            "center_x": center.get("center_x"),
            "center_y": center.get("center_y"),
            "center_z": center.get("center_z"),
            "size_x": size.get("size_x"),
            "size_y": size.get("size_y"),
            "size_z": size.get("size_z"),
            "threads": threads,
            "ligand_index": i
        }

# --------------------- Job execution ---------------------
def run_job(job, vina_bin, gpu_id):
    """Run QuickVina2-GPU for one ligand."""
    receptor = job["receptor"]
    ligand = job["ligand"]
    output_file = job["output_file"]

    out_dir = Path(output_file).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = out_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{Path(ligand).stem}_gpu{gpu_id}.log"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = (
        f"{ENV_SETUP}; {vina_bin} "
        f"--receptor {receptor} "
        f"--ligand {ligand} "
        f"--out {output_file} "
        f"--center_x {job['center_x']} --center_y {job['center_y']} --center_z {job['center_z']} "
        f"--size_x {job['size_x']} --size_y {job['size_y']} --size_z {job['size_z']} "
        f"--thread {job['threads']}"
    )

    start = time.time()
    subprocess.call(cmd, shell=True, stdout=open(log_file, "w"),
                    stderr=subprocess.STDOUT, executable="/bin/bash")
    end = time.time()

    return {
        **job,
        "gpu_id": gpu_id,
        "log_file": str(log_file.resolve()),
        "runtime_s": round(end - start, 2)
    }

# --------------------- Main ---------------------
def main():
    args = parse_args()
    df = pd.read_csv(args.csv)

    # Gather jobs
    configs = []
    for _, row in df.iterrows():
        for c in ("config_actives", "config_inactives"):
            if row.get(c) and Path(row[c]).exists():
                configs.append(row[c])

    all_jobs = []
    for cfg in configs:
        all_jobs.extend(list(expand_config(cfg, args.threads, test_run=args.test_run)))

    if args.dry_run:
        all_jobs = all_jobs[:args.dry_run]

    print(f"📋 Prepared {len(all_jobs)} ligand jobs from {len(configs)} configs.")

    # Run jobs sequentially
    manifests = {}
    files = {}
    with tqdm(total=len(all_jobs), desc="Docking", unit="ligand") as pbar:
        for job in all_jobs:
            result = run_job(job, str(Path(args.vina_bin).resolve()), args.gpu_id)

            out_file = Path(result["output_file"])
            manifest_path = out_file.parent / "manifest.csv"
            if manifest_path not in manifests:
                f = open(manifest_path, "w", newline="")
                files[manifest_path] = f
                writer = csv.DictWriter(f, fieldnames=list(result.keys()))
                writer.writeheader()
                manifests[manifest_path] = writer

            manifests[manifest_path].writerow(result)
            files[manifest_path].flush()
            pbar.update(1)

    for f in files.values():
        f.close()

    print("✅ Docking complete. Manifests written per output folder.")

if __name__ == "__main__":
    main()
