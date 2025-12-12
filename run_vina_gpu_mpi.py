#!/usr/bin/env python3
"""
run_vina_gpu_mpi.py

A minimal, robust MPI work-queue for running QuickVina2-GPU
(one ligand per job) with per-GPU worker ranks.

Key points:
- Build the job list on rank 0 only (config expansion is cheap I/O).
- Pull-based master/worker (READY/START/DONE/STOP) for perfect load balance.
- One worker process per GPU; GPU bound via local rank or CUDA_VISIBLE_DEVICES index.
- No pandas/tqdm dependency; optional single-line progress from rank 0.
- Manifest written incrementally (global + per-target), O(#workers) memory.
- Logs contain **only Vina's stdout/stderr** for each ligand.
- Workers don't chdir per job; we resolve the binary once and run with cwd=bin_dir.

Run (example with Slurm):
    mpirun -np 3 --oversubscribe --map-by ppr:3:node --bind-to none \
    python run_vina_gpu_mpi.py --smoke --smoke_n 50 --threads 0 --progress

Manifest/logs:
- <output_dir>/log/<ligand_name>_gpu<GPU_ID>.log
- docking_manifest.csv (one row per ligand with status/rc/log path)
"""

import argparse, os, csv, subprocess, shutil, numpy as np
from pathlib import Path
from mpi4py import MPI
import sys, time
from typing import Optional

# Prefer preloading modules in the job script
REQUIRED_COLS = [
    "target", "receptor_pdbqt",
    "actives_dir", "inactives_dir",
    "docked_vina_actives", "docked_vina_inactives",
    "center_x", "center_y", "center_z",
    "size_x", "size_y", "size_z",
]


# --------------------- CLI ---------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="LIT_PCBA/vina_boxes.csv",
                    help="Path to vina_boxes.csv (default: LIT_PCBA/vina_boxes.csv)")
    ap.add_argument("--vina_bin", default="vina-gpu-dev/QuickVina2-GPU-2-1",
                    help="Path or name of QuickVina2-GPU binary (default: vina-gpu-dev/QuickVina2-GPU-2-1)")
    ap.add_argument("--gpus", type=int, default=None,
                    help="GPUs per node (default: autodetect via nvidia-smi)")
    ap.add_argument("--threads", type=int, default=8000,
                    help="Threads per job. Use 0 to auto-tune per GPU (default: 8000)")
    ap.add_argument("--seed", default="0",
                    help='--seed passed to Vina (default: "0")')
    ap.add_argument("--dry_run", type=int, default=0,
                    help="If >0, limit to this many ligands total")
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

def _read_numpy_csv(csv_path: Path):
    """
    Load CSV with NumPy structured array (names=True); returns a 1D structured array.
    Handles the 0-dim corner case when there's a single data row.
    """
    arr = np.genfromtxt(
        csv_path,
        delimiter=",",
        names=True,
        dtype=None,            # infer dtypes; strings become <U* (unicode)
        encoding="utf-8",
    )
    # If only one row, genfromtxt returns a 0-d array; coerce to 1-d
    if arr.shape == ():
        arr = np.array([arr], dtype=arr.dtype)
    return arr

def auto_threads_for_gpu() -> int:
    """
    Choose a sane --thread for QVina2-GPU based on GPU properties.
    Heuristic: 256 lanes per SM, cap at 9000 (QVina2-GPU sweet spot).
    Falls back to 8000 if probing fails.
    """
    try:
        out = subprocess.check_output([
            "bash", "-lc",
            "nvidia-smi --query-gpu=multiprocessors --format=csv,noheader,nounits | head -n 1"
        ], text=True).strip()
        sms = int(out)
        threads = min(9000, max(3000, 256 * sms))
        return threads
    except Exception:
        return 8000

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
    for lig in sorted(lig_dir.glob("*.pdbqt")):
        out_file = out_dir / f"{lig.stem}_docked.pdbqt"
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
    # Validate header strictly (prevents silent mis-parsing)
    with open(csv_path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
    missing = [c for c in REQUIRED_COLS if c not in header]
    if missing:
        raise ValueError(f"vina_boxes.csv missing required columns: {missing}")

    arr = _read_numpy_csv(csv_path)

    # Fail fast if any numeric fields are non-floatable
    try:
        cx = arr["center_x"].astype(float)
        cy = arr["center_y"].astype(float)
        cz = arr["center_z"].astype(float)
        sx = arr["size_x"].astype(float)
        sy = arr["size_y"].astype(float)
        sz = arr["size_z"].astype(float)
    except Exception as e:
        raise ValueError(f"Non-numeric value in box columns: {e}")

    # Vector-ish path resolution (per row)
    targets   = arr["target"]
    receptors = np.array([str(Path(p).resolve()) for p in arr["receptor_pdbqt"]], dtype=object)

    act_in    = np.array([Path(p) for p in arr["actives_dir"]], dtype=object)
    inact_in  = np.array([Path(p) for p in arr["inactives_dir"]], dtype=object)
    act_out   = np.array([Path(p) for p in arr["docked_vina_actives"]], dtype=object)
    inact_out = np.array([Path(p) for p in arr["docked_vina_inactives"]], dtype=object)

    # Create output/log dirs once per unique path
    for p in np.unique(np.concatenate([act_out, inact_out]).astype(object)):
        Path(p).mkdir(parents=True, exist_ok=True)
        (Path(p) / "log").mkdir(parents=True, exist_ok=True)

    jobs = []
    # Expand ligands per row (filesystem expansion is inherently iterative)
    for i in range(arr.shape[0]):
        # Actives
        ligs = sorted(act_in[i].glob("*.pdbqt"))
        for lig in ligs:
            ligand_name = lig.stem
            out_file = act_out[i] / f"{ligand_name}_docked.pdbqt"
            jobs.append({
                "receptor":   receptors[i],
                "ligand":     str(lig.resolve()),
                "output_file": str(out_file.resolve()),
                "center_x": float(cx[i]), "center_y": float(cy[i]), "center_z": float(cz[i]),
                "size_x":   float(sx[i]), "size_y":   float(sy[i]), "size_z":   float(sz[i]),
                "threads": threads,  # 0 allowed; worker auto-tunes
                "ligand_name": ligand_name,
                "target": str(targets[i]),
            })
            if limit and len(jobs) >= limit:
                return jobs[:limit]

        # Inactives
        ligs = sorted(inact_in[i].glob("*.pdbqt"))
        for lig in ligs:
            ligand_name = lig.stem
            out_file = inact_out[i] / f"{ligand_name}_docked.pdbqt"
            jobs.append({
                "receptor":   receptors[i],
                "ligand":     str(lig.resolve()),
                "output_file": str(out_file.resolve()),
                "center_x": float(cx[i]), "center_y": float(cy[i]), "center_z": float(cz[i]),
                "size_x":   float(sx[i]), "size_y":   float(sy[i]), "size_z":   float(sz[i]),
                "threads": threads,
                "ligand_name": ligand_name,
                "target": str(targets[i]),
            })
            if limit and len(jobs) >= limit:
                return jobs[:limit]

    return jobs
#--------------- Runner ---------------------

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

def run_job(job: dict, vina_bin: str, gpu_id: int, base_env: dict, bin_dir: Optional[str], seed: str):
    out_file = Path(job["output_file"])
    out_dir = out_file.parent
    log_dir = out_dir / "log"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job['ligand_name']}_gpu{gpu_id}.log"

    # Clone the prebuilt worker env so we can tweak per job if needed
    env = base_env.copy()
    vina_path = vina_bin

    # Build argv (seed only applied here)
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
        "--thread", str(job["threads"]),
        "--seed", str(seed),
    ]

    # Execute: keep logs as Vina stdout/stderr only; no per-job env_setup/prologue
    with open(log_file, "w") as lf:
        rc = subprocess.call(args, stdout=lf, stderr=subprocess.STDOUT, env=env,
                             cwd=bin_dir if bin_dir else None)

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
        for _t, (fh, _w, _p) in list(self.cache.items()):
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

    # Per-target tracking (for minimal prints): totals, started flags, done counts, start timestamps
    target_total = {}
    for j in jobs:
        t = j.get("target") or "unknown"
        target_total[t] = target_total.get(t, 0) + 1
    target_done = {t: 0 for t in target_total}
    target_started = set()
    target_start_ts = {}

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
        # Periodic progress is off by default for ALL modes; enable only with --progress
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
        sys.stdout.write("" + msg)
        sys.stdout.flush()
        last_print = now

    while closed_workers < size - 1:
        status = MPI.Status()
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()

        if tag == READY:
            if job_idx < n_total:
                # Print a single START message per target when its first ligand is dispatched
                tgt = jobs[job_idx].get("target") or "unknown"
                if tgt not in target_started and not args.quiet:
                    target_started.add(tgt)
                    target_start_ts[tgt] = time.time()
                    print(f"[target] START  {tgt}  ({target_total.get(tgt, 0)} ligands)")
                comm.send(jobs[job_idx], dest=source, tag=START)
                job_idx += 1
            else:
                comm.send(None, dest=source, tag=STOP)
        elif tag == DONE:
            res = data
            if res:
                writer.writerow(res)
                t = (res.get("target") or "unknown").replace("/", "_")
                # Update per-target done count and emit a single END message when finished
                tgt_raw = res.get("target") or "unknown"
                if tgt_raw in target_done:
                    target_done[tgt_raw] += 1
                    if target_done[tgt_raw] == target_total.get(tgt_raw, 0):
                        if not args.quiet:
                            dt = time.time() - target_start_ts.get(tgt_raw, time.time())
                            print(f"[target] DONE   {tgt_raw}  ({target_total.get(tgt_raw, 0)} ligands) in {dt:.1f}s")
                tw = wc.get(t)
                tw.writerow(res)
            if not args.quiet and res and (res.get("return_code", 0) != 0):
                print(f"[master] job error target={res.get('target')} ligand={res.get('ligand_name')} rc={res.get('return_code')}")
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

    # Resolve the binary once and prepare a base environment once
    vina_bin_resolved = resolve_vina_bin(args.vina_bin)
    bin_path = Path(vina_bin_resolved)
    bin_dir = str(bin_path.parent) if bin_path.exists() else None
    base_env = os.environ.copy()
    base_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if bin_dir:
        base_env["LD_LIBRARY_PATH"] = (bin_dir + ":" + base_env.get("LD_LIBRARY_PATH", "")).rstrip(":")

    # If auto-threads requested, compute once per worker/GPU
    auto_threads = None
    if args.threads == 0:
        auto_threads = auto_threads_for_gpu()

    while True:
        comm.send(None, dest=0, tag=READY)
        status = MPI.Status()
        job = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag == START and job is not None:
            if auto_threads is not None:
                job["threads"] = auto_threads
            result = run_job(job, vina_bin_resolved, gpu_id, base_env, bin_dir, seed=args.seed)
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
