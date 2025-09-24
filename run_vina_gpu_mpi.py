#!/usr/bin/env python3
"""
run_vina_gpu_mpi.py

A minimal, robust MPI work-queue for running QuickVina2‑GPU
(one ligand per job) with per‑GPU worker ranks.

Key points:
- Build the job list on rank 0 only (config expansion is cheap I/O).
- Pull‑based master/worker (READY/START/DONE/STOP) for perfect load balance.
- One worker process per GPU; GPU bound via local rank or CUDA_VISIBLE_DEVICES index.
- No pandas/tqdm dependency; optional single-line progress from rank 0.
- Manifest written incrementally (global + per-target), O(#workers) memory.
- Logs contain **only Vina's stdout/stderr** for each ligand.
- Workers **cd into the Vina binary directory** (like your smoketest) so kernels and local libs are found.

Run (example with Slurm):
  srun --mpi=pmix --gpus-per-task=1 -n <num_gpus_total> python run_vina_gpu_mpi.py \
      --csv LIT_PCBA/vina_boxes.csv --vina_bin vina-gpu-dev/QuickVina2-GPU-2-1

Manifest/logs:
- <output_dir>/log/<ligand_name>_gpu<GPU_ID>.log
- docking_manifest.csv (one row per ligand with status/rc/log path)
"""

import argparse, os, csv, shlex, subprocess, shutil
from pathlib import Path
from mpi4py import MPI
import sys, time

# Default ENV setup (cluster modules)
ENV_SETUP = (
    "ulimit -s 8192 || true; "
    "module purge || true; "
    "module load Boost/1.77.0-GCC-11.2.0  Python/3.9.6-GCCcore-11.2.0 "
    "CUDA/12.0.0  OpenMPI/4.1.1-GCC-11.2.0 || true"
)

# --------------------- CLI ---------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="LIT_PCBA/vina_boxes.csv",
                    help="Path to vina_boxes.csv (default: LIT_PCBA/vina_boxes.csv)")
    ap.add_argument("--vina_bin", default="vina-gpu-dev/QuickVina2-GPU-2-1",
                    help="Path or name of QuickVina2‑GPU binary (default: vina-gpu-dev/QuickVina2-GPU-2-1)")
    ap.add_argument("--gpus", type=int, default=None,
                    help="GPUs per node (default: autodetect via nvidia-smi)")
    ap.add_argument("--threads", type=int, default=8000,
                    help="--thread passed to Vina per job (default: 8000)")
    ap.add_argument("--dry_run", type=int, default=0,
                    help="If >0, limit to this many ligands total")
    ap.add_argument("--env_setup", default=ENV_SETUP,
                    help="Optional shell prefix for env/modules (default loads Boost/Python/CUDA/OpenMPI)")
    ap.add_argument("--quiet", action="store_true",
                    help="Less verbose output on rank 0")
    # Smoke-test mode (does NOT change threads; only limits ligands)
    ap.add_argument("--smoke", action="store_true",
                    help="Enable a tiny end-to-end test that limits ligands only")
    ap.add_argument("--smoke_n", type=int, default=16,
                    help="If --smoke, total ligands to run (default 16)")
    # Cap open file descriptors for per-target manifests
    ap.add_argument("--manifest_fd_cap", type=int, default=128,
                    help="Max concurrently open per-target manifest files (LRU closed)")
    # Lightweight progress options
    ap.add_argument("--progress", action="store_true",
                    help="Show a lightweight single-line progress bar on rank 0")
    ap.add_argument("--progress_every", type=float, default=1.0,
                    help="Seconds between progress updates (default 1.0)")
    return ap.parse_args()

# --------------------- GPU helpers ---------------------

def detect_num_gpus():
    try:
        out = subprocess.check_output(["bash", "-lc", "nvidia-smi -L | wc -l"], text=True)
        n = int(out.strip())
        return n if n > 0 else 1
    except Exception:
        return 1


def pick_gpu_id(num_gpus: int) -> int:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis and "," not in vis:
        # single device exposed -> always 0
        return 0
    # decide by local rank within the node
    local_rank = int(
        os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
        or os.environ.get("PMI_LOCAL_RANK")
        or os.environ.get("SLURM_LOCALID", "0")
    )
    if vis:
        ids = [x.strip() for x in vis.split(",") if x.strip()]
        return int(ids[local_rank % len(ids)])
    return local_rank % max(1, num_gpus)

# --------------------- Config expansion ---------------------

def expand_config(cfg_path: Path, threads: int):
    cfg_path = Path(cfg_path)
    receptor = lig_dir = out_dir = None
    center, size = {}, {}
    with open(cfg_path) as f:
        for raw in f:
            line = raw.strip()
            if not line or "=" not in line:
                continue
            k, v = [x.strip() for x in line.split("=", 1)]
            if k == "receptor":
                receptor = Path(v)
                if not receptor.is_absolute():
                    receptor = cfg_path.parent / receptor
            elif k == "ligand_directory":
                lig_dir = Path(v)
                if not lig_dir.is_absolute():
                    lig_dir = cfg_path.parent / lig_dir
            elif k in ("out", "output_directory"):
                out_dir = Path(v)
                if not out_dir.is_absolute():
                    out_dir = cfg_path.parent / out_dir
            elif k.startswith("center_"):
                center[k] = float(v)
            elif k.startswith("size_"):
                size[k] = float(v)
    if not (receptor and lig_dir and out_dir):
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, lig in enumerate(sorted(lig_dir.glob("*.pdbqt"))):
        out_file = out_dir / f"{lig.stem}_out.pdbqt"
        jobs.append({
            "receptor": str(receptor.resolve()),
            "ligand": str(lig.resolve()),
            "output_file": str(out_file.resolve()),
            "center_x": center.get("center_x"),
            "center_y": center.get("center_y"),
            "center_z": center.get("center_z"),
            "size_x": size.get("size_x"),
            "size_y": size.get("size_y"),
            "size_z": size.get("size_z"),
            "threads": threads,
            "ligand_name": lig.stem,
        })
    return jobs


def build_jobs_from_boxes(csv_path: Path, threads: int, limit: int = 0):
    csv_path = Path(csv_path)
    configs = []  # list of (config_path, target)
    with open(csv_path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            target = row.get("target") or row.get("protein") or row.get("name") or "unknown"
            for col in ("config_actives", "config_inactives"):
                p = row.get(col)
                if p:
                    configs.append((Path(p), target))

    jobs = []
    for cfg, target in configs:
        for job in expand_config(cfg, threads):
            job["target"] = target
            jobs.append(job)
            if limit and len(jobs) >= limit:
                return jobs[:limit]
    return jobs

# --------------------- Runner ---------------------

READY, START, DONE, STOP = 1, 2, 3, 4


def resolve_vina_bin(user_bin: str) -> str:
    """Resolve QuickVina2-GPU path (like the smoketest).
    Returns absolute path if found, else the original string (to allow PATH/module resolution).
    """
    p = Path(user_bin)
    if p.exists():
        return str(p.resolve())
    if p.is_dir():
        cand = p / "QuickVina2-GPU-2-1"
        if cand.exists():
            return str(cand.resolve())
    for cand in (
        Path("./vina-gpu-dev/QuickVina2-GPU-2-1/QuickVina2-GPU-2-1"),
        Path("./vina-gpu-dev/QuickVina2-GPU-2-1"),
        Path("./QuickVina2-GPU-2-1"),
    ):
        if cand.exists():
            return str(cand.resolve())
    which = shutil.which("QuickVina2-GPU-2-1")
    if which:
        return which
    return user_bin


def run_job(job: dict, vina_bin: str, gpu_id: int, env_setup: str = ""):
    out_file = Path(job["output_file"])    
    out_dir = out_file.parent
    log_dir = out_dir / "log"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job['ligand_name']}_gpu{gpu_id}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Resolve binary (absolute if local), prepend its dir to LD_LIBRARY_PATH
    vina_path = resolve_vina_bin(vina_bin)
    bin_dir = str(Path(vina_path).parent) if os.path.isabs(vina_path) and os.path.exists(vina_path) else None
    if bin_dir:
        env["LD_LIBRARY_PATH"] = (bin_dir + ":" + env.get("LD_LIBRARY_PATH", "")).rstrip(":")

    # Build argv
    args = [
        vina_path,
        "--receptor", job["receptor"],
        "--ligand", job["ligand"],
        "--out", str(out_file),
        "--center_x", str(job["center_x"]),
        "--center_y", str(job["center_y"]),
        "--center_z", str(job["center_z"]),
        "--size_x", str(job["size_x"]),
        "--size_y", str(job["size_y"]),
        "--size_z", str(job["size_z"]),
        "--thread", str(job["threads"]) ,
    ]

    # Execute: we already chdir'ed into bin_dir at worker start; keep logs as Vina stdout/stderr only
    if env_setup:
        vina_cmd = " ".join(shlex.quote(x) for x in args)
        setup = env_setup.strip().rstrip(';')
        prologue = (
            "set -euo pipefail; "
            "if type module >/dev/null 2>&1; then :; "
            "elif [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; "
            "elif [ -n \"$MODULESHOME\" ] && [ -f \"$MODULESHOME/init/bash\" ]; then . \"$MODULESHOME/init/bash\"; fi; "
        )
        shell_cmd = f"{{ {prologue} {setup} ; }} >/dev/null 2>&1; exec {vina_cmd}"
        with open(log_file, "w") as lf:
            rc = subprocess.call(shell_cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT,
                                  executable="/bin/bash", env=env)
    else:
        with open(log_file, "w") as lf:
            rc = subprocess.call(args, stdout=lf, stderr=subprocess.STDOUT, env=env)

    return {
        **job,
        "gpu_id": gpu_id,
        "log_file": str(log_file.resolve()),
        "return_code": rc,
        "status": "ok" if rc == 0 else "error",
    }

# --------------------- Master / Worker ---------------------

# LRU cache for per-target manifest writers
class WriterCache:
    def __init__(self, base_dir: Path, fieldnames, cap: int):
        self.base_dir = Path(base_dir)
        self.fieldnames = list(fieldnames)
        self.cap = max(8, cap)
        self.cache = {}  # target -> (fh, writer)
        self.order = []  # LRU order oldest..newest

    def _open_writer(self, target: str):
        tpath = self.base_dir / f"docking_manifest_{target}.csv"
        fh = open(tpath, "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=self.fieldnames)
        if tpath.stat().st_size == 0:
            writer.writeheader()
        return fh, writer, tpath

    def get(self, target: str):
        if target in self.cache:
            if target in self.order:
                self.order.remove(target)
            self.order.append(target)
            return self.cache[target][1]
        if len(self.cache) >= self.cap:
            old = self.order.pop(0)
            fh, _w, _p = self.cache.pop(old)
            try:
                fh.close()
            except Exception:
                pass
        fh, writer, tpath = self._open_writer(target)
        self.cache[target] = (fh, writer, tpath)
        self.order.append(target)
        return writer

    def close_all(self):
        for t, (fh, _w, _p) in list(self.cache.items()):
            try:
                fh.close()
            except Exception:
                pass
        self.cache.clear()
        self.order.clear()


def master(comm: MPI.Comm, args):
    rank = comm.Get_rank()
    assert rank == 0

    limit = args.dry_run or (args.smoke_n if getattr(args, 'smoke', False) else 0)
    jobs = build_jobs_from_boxes(Path(args.csv), args.threads, limit)
    n_total = len(jobs)
    if not args.quiet:
        print(f"[master] total jobs: {n_total}")

    # Global manifest
    manifest_path = Path(args.csv).parent / "docking_manifest.csv"
    mf = open(manifest_path, "w", newline="")
    base_fields = list(jobs[0].keys()) if jobs else []
    fieldnames = base_fields + ["gpu_id", "log_file", "return_code", "status"]
    writer = csv.DictWriter(mf, fieldnames=fieldnames)
    if jobs:
        writer.writeheader()

    # Per-target manifests via LRU cache
    wc = WriterCache(Path(args.csv).parent, fieldnames, getattr(args, 'manifest_fd_cap', 128))

    # active workers count
    size = comm.Get_size()
    closed_workers = 0
    job_idx = 0
    done = 0
    start_ts = time.time()
    last_print = 0.0

    def maybe_progress(force=False):
        nonlocal last_print
        if args.quiet or not getattr(args, 'progress', False) or n_total == 0:
            return
        now = time.time()
        if not force and (now - last_print) < getattr(args, 'progress_every', 1.0):
            return
        pct = (done / n_total * 100.0) if n_total else 100.0
        rate = done / max(1e-9, (now - start_ts))
        remaining = n_total - done
        eta = remaining / rate if rate > 0 else 0
        msg = f"[Docking] {done}/{n_total} ({pct:5.1f}%) | {rate:6.2f} lig/s | ETA {eta:6.1f}s"
        sys.stdout.write("\r" + msg)
        sys.stdout.flush()
        last_print = now

    while closed_workers < size - 1:
        status = MPI.Status()
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()

        if tag == READY:
            if job_idx < n_total:
                comm.send(jobs[job_idx], dest=source, tag=START)
                job_idx += 1
            else:
                comm.send(None, dest=source, tag=STOP)
        elif tag == DONE:
            res = data
            if res:
                writer.writerow(res)
                t = (res.get("target") or "unknown").replace("/", "_")
                tw = wc.get(t)
                tw.writerow(res)
            if not args.quiet and res and (res.get("return_code", 0) != 0):
                print(f"\n[master] job error target={res.get('target')} ligand={res.get('ligand_name')} rc={res.get('return_code')}")
            done += 1
            maybe_progress()
        elif tag == STOP:
            closed_workers += 1

    mf.close()
    wc.close_all()
    if getattr(args, 'progress', False) and not args.quiet:
        maybe_progress(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()
    if not args.quiet:
        print(f"[master] manifest -> {manifest_path}")


def worker(comm: MPI.Comm, args):
    num_gpus = args.gpus or detect_num_gpus()
    gpu_id = pick_gpu_id(num_gpus)

    # Resolve the binary and chdir into its directory (like smoketest)
    vina_bin_resolved = resolve_vina_bin(args.vina_bin)
    bin_path = Path(vina_bin_resolved)
    if bin_path.exists():
        os.chdir(str(bin_path.parent))

    while True:
        comm.send(None, dest=0, tag=READY)
        status = MPI.Status()
        job = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag == START and job is not None:
            result = run_job(job, vina_bin_resolved, gpu_id, env_setup=args.env_setup)
            comm.send(result, dest=0, tag=DONE)
        elif tag == STOP:
            comm.send(None, dest=0, tag=STOP)
            break

# --------------------- Main ---------------------

def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        master(comm, args)
    else:
        worker(comm, args)

if __name__ == "__main__":
    main()
