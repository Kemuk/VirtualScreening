#!/usr/bin/env python3
"""
run_vina_gpu_mpi.py

MPI-based parallel docking with QuickVina2-GPU:
1. Parallel creation of vina_jobs.csv from vina_boxes.csv (one job per ligand).
2. Parallel execution of jobs across GPUs.

- Each job = one ligand file.
- Outputs written directly to the exact output_file path.
- Logs written into <output_dir>/log/<ligand_name>_gpuX.log.
- docking_manifest.csv tracks one row per ligand.
"""

import argparse, os, subprocess, csv
from pathlib import Path
import pandas as pd
from mpi4py import MPI
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
                    help="Original vina_boxes.csv file")
    ap.add_argument("--vina_bin", default="vina-gpu-dev/QuickVina2-GPU-2-1")
    ap.add_argument("--gpus", type=int, default=None,
                    help="Number of GPUs per node (default: auto-detect)")
    ap.add_argument("--threads", type=int, default=8000,
                    help="Threads per job (default: 8000)")
    ap.add_argument("--dry_run", type=int, default=0,
                    help="If >0, only run this many ligands for testing")
    return ap.parse_args()

def detect_num_gpus():
    try:
        out = subprocess.check_output("nvidia-smi -L | wc -l", shell=True)
        return int(out.decode().strip())
    except Exception:
        return 1

def pick_gpu_id(num_gpus):
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis and "," not in vis:
        return 0
    local_rank = int(
        os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
        or os.environ.get("PMI_LOCAL_RANK")
        or os.environ.get("SLURM_LOCALID", "0")
    )
    if vis:
        ids = [x.strip() for x in vis.split(",") if x.strip() != ""]
        return int(ids[local_rank % len(ids)])
    return local_rank % num_gpus

# --------------------- Job expansion ---------------------
def expand_config(cfg_path, threads):
    """Expand one config file into per-ligand job definitions."""
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
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    ligands = sorted(lig_dir.glob("*.pdbqt"))

    jobs = []
    for i, lig in enumerate(ligands):
        output_file = out_dir / f"{lig.stem}_out.pdbqt"
        jobs.append({
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
        })
    return jobs

# --------------------- Job execution ---------------------
def run_job(job, vina_bin, gpu_id):
    """Run QuickVina2-GPU on one ligand job."""
    receptor = job["receptor"]
    ligand = job["ligand"]
    output_file = job["output_file"]

    out_dir = Path(output_file).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # log file
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

    subprocess.call(cmd, shell=True, stdout=open(log_file, "w"),
                    stderr=subprocess.STDOUT, executable="/bin/bash")

    return {
        **job,
        "gpu_id": gpu_id,
        "log_file": str(log_file.resolve())
    }

# --------------------- Manager ---------------------
class DockingManager:
    def __init__(self, args, comm, rank, size):
        self.args = args
        self.comm = comm
        self.rank = rank
        self.size = size
        self.vina_bin = Path(args.vina_bin).resolve()
        self.num_gpus = args.gpus or detect_num_gpus()
        self.jobs = []
        self.results = []

    def build_jobs_parallel(self):
        if self.rank == 0:
            df = pd.read_csv(self.args.csv)
            configs = []
            for _, row in df.iterrows():
                for c in ("config_actives", "config_inactives"):
                    if row.get(c) and Path(row[c]).exists():
                        configs.append(row[c])

            for i, cfg in enumerate(configs):
                self.comm.send(cfg, dest=(i % (self.size - 1)) + 1, tag=10)

            jobs = []
            for _ in configs:
                jobs.extend(self.comm.recv(source=MPI.ANY_SOURCE, tag=11))

            for r in range(1, self.size):
                self.comm.send("STOP", dest=r, tag=10)

            jobs_csv = Path(self.args.csv).parent / "vina_jobs.csv"
            with open(jobs_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=jobs[0].keys())
                writer.writeheader()
                for job in jobs:
                    writer.writerow(job)
            print(f"vina_jobs.csv written to {jobs_csv}")
            self.jobs = jobs
        else:
            while True:
                cfg = self.comm.recv(source=0, tag=MPI.ANY_TAG)
                if cfg == "STOP":
                    break
                jobs = expand_config(cfg, self.args.threads)
                self.comm.send(jobs, dest=0, tag=11)

    def run_master(self):
        for i, job in enumerate(self.jobs):
            if self.args.dry_run and i >= self.args.dry_run:
                break
            self.comm.send(job, dest=(i % (self.size - 1)) + 1, tag=20)

        n_jobs = min(len(self.jobs), self.args.dry_run or len(self.jobs))
        with tqdm(total=n_jobs, desc="Docking", unit="ligand") as pbar:
            for _ in range(n_jobs):
                result = self.comm.recv(source=MPI.ANY_SOURCE, tag=21)
                self.results.append(result)
                pbar.update(1)

        for r in range(1, self.size):
            self.comm.send("STOP", dest=r, tag=20)

        self.write_manifest()

    def run_worker(self):
        gpu_id = pick_gpu_id(self.num_gpus)
        while True:
            task = self.comm.recv(source=0, tag=MPI.ANY_TAG)
            if task == "STOP":
                break
            result = run_job(task, str(self.vina_bin), gpu_id)
            self.comm.send(result, dest=0, tag=21)

    def write_manifest(self):
        manifest_path = Path(self.args.csv).parent / "docking_manifest.csv"
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.results[0].keys()))
            writer.writeheader()
            for row in self.results:
                writer.writerow(row)
        print(f"Docking manifest written to {manifest_path}")

# --------------------- Main ---------------------
def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    manager = DockingManager(args, comm, rank, size)

    # Phase 1: parallel job building
    manager.build_jobs_parallel()

    # Phase 2: docking
    if rank == 0:
        manager.run_master()
    else:
        manager.run_worker()

if __name__ == "__main__":
    main()
