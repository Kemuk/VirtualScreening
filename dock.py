#!/usr/bin/env python3
"""
Unified docking runner. Default CPU (multiprocessing + vina).
Use --mode gpu to run QuickVina2-GPU sequentially on one GPU.

Example:
  python dock.py --csv LIT_PCBA/vina_boxes.csv --mode cpu --cpu 4 --workers 8
  python dock.py --csv LIT_PCBA/vina_boxes.csv --mode gpu --gpu_id 0 --threads 8000
"""
from pathlib import Path
import argparse, csv, os, subprocess, shutil, time, hashlib
import multiprocessing
from multiprocessing import Pool, Lock
import numpy as np

MANIFEST_LOCK = Lock()
REQUIRED_COLS = [
    "target","receptor_pdbqt","actives_dir","inactives_dir",
    "docked_vina_actives","docked_vina_inactives",
    "center_x","center_y","center_z","size_x","size_y","size_z",
]

ENV_SETUP = (
    "ulimit -s 8192 || true; "
    "module purge || true; "
    "module load Boost/1.77.0-GCC-11.2.0  Python/3.9.6-GCCcore-11.2.0 "
    "CUDA/12.0.0  OpenMPI/4.1.1-GCC-11.2.0 || true"
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="LIT_PCBA/vina_boxes.csv")
    p.add_argument("--vina_bin", default="vina")
    p.add_argument("--mode", choices=("cpu","gpu"), default="cpu", help="cpu (default) or gpu")
    p.add_argument("--cpu", type=int, default=1, help="CPUs per vina job (cpu mode)")
    p.add_argument("--workers", type=int, default=0, help="Number of worker processes (cpu mode). 0 => auto")
    p.add_argument("--threads", type=int, default=8000, help="Threads per GPU job (gpu mode)")
    p.add_argument("--gpu_id", type=int, default=0, help="GPU id (gpu mode)")
    p.add_argument("--dry_run", type=int, default=0, help="If >0, limit total jobs to this")
    p.add_argument("--test_run", action="store_true", help="Only first ligand per box")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--shard_idx", type=int, default=-1, help="array task index (0-based).")
    p.add_argument("--shard_total", type=int, default=1, help="number of array tasks.")
    return p.parse_args()

def _read_numpy_csv(csv_path: Path):
    a = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if a.shape == (): a = np.array([a], dtype=a.dtype)
    return a

def build_jobs_from_boxes(csv_path: Path, cpu_threads:int, gpu_threads:int, limit=0, test_run=False):
    with open(csv_path, newline="") as f:
        rdr = csv.reader(f); header = next(rdr)
    missing = [c for c in REQUIRED_COLS if c not in header]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    arr = _read_numpy_csv(csv_path)
    cx = arr["center_x"].astype(float); cy = arr["center_y"].astype(float); cz = arr["center_z"].astype(float)
    sx = arr["size_x"].astype(float); sy = arr["size_y"].astype(float); sz = arr["size_z"].astype(float)
    targets = arr["target"]
    receptors = [str(Path(p).resolve()) for p in arr["receptor_pdbqt"]]
    act_in = np.array([Path(p) for p in arr["actives_dir"]], dtype=object)
    inact_in = np.array([Path(p) for p in arr["inactives_dir"]], dtype=object)
    act_out = np.array([Path(p) for p in arr["docked_vina_actives"]], dtype=object)
    inact_out = np.array([Path(p) for p in arr["docked_vina_inactives"]], dtype=object)

    for p in set(list(act_out)+list(inact_out)):
        Path(p).mkdir(parents=True, exist_ok=True); (Path(p)/"log").mkdir(parents=True, exist_ok=True)

    jobs=[]
    for i in range(arr.shape[0]):
        # actives
        for lig in sorted(act_in[i].glob("*.pdbqt")):
            ln = lig.stem
            out = act_out[i] / f"{ln}_docked.pdbqt"
            jobs.append({
                "target": str(targets[i]), "receptor": receptors[i],
                "ligand": str(lig.resolve()), "ligand_name": ln,
                "output_file": str(out.resolve()),
                "center_x": float(cx[i]), "center_y": float(cy[i]), "center_z": float(cz[i]),
                "size_x": float(sx[i]), "size_y": float(sy[i]), "size_z": float(sz[i]),
                "cpu": cpu_threads, "threads": gpu_threads,
                "num_modes": 1, "exhaustiveness": 8, "seed": 42
            })
            if limit and len(jobs)>=limit: return jobs[:limit]
            if test_run: break
        # inactives
        for lig in sorted(inact_in[i].glob("*.pdbqt")):
            ln = lig.stem
            out = inact_out[i] / f"{ln}_docked.pdbqt"
            jobs.append({
                "target": str(targets[i]), "receptor": receptors[i],
                "ligand": str(lig.resolve()), "ligand_name": ln,
                "output_file": str(out.resolve()),
                "center_x": float(cx[i]), "center_y": float(cy[i]), "center_z": float(cz[i]),
                "size_x": float(sx[i]), "size_y": float(sy[i]), "size_z": float(sz[i]),
                "cpu": cpu_threads, "threads": gpu_threads,
                "num_modes": 1, "exhaustiveness": 8, "seed": 42
            })
            if limit and len(jobs)>=limit: return jobs[:limit]
            if test_run: break
    return jobs

def resolve_vina_bin(user_bin:str)->str:
    p = Path(user_bin)
    if p.exists(): return str(p.resolve())
    w = shutil.which(user_bin); return w if w else user_bin

def stable_hash_mod(s:str, mod:int)->int:
    h = hashlib.md5(s.encode('utf-8')).hexdigest()
    return int(h,16) % mod

def run_job_cpu(job, vina_bin):
    ligand = job["ligand_name"]
    out_file = Path(job["output_file"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    arr_id = os.getenv("SLURM_ARRAY_TASK_ID","")
    log_file = out_file.parent / "log" / (f"{ligand}_A{arr_id}.log" if arr_id else f"{ligand}.log")
    cmd = [
        vina_bin, "--receptor", job["receptor"], "--ligand", job["ligand"],
        "--center_x", str(job["center_x"]), "--center_y", str(job["center_y"]), "--center_z", str(job["center_z"]),
        "--size_x", str(job["size_x"]), "--size_y", str(job["size_y"]), "--size_z", str(job["size_z"]),
        "--exhaustiveness", str(job.get("exhaustiveness",8)),
        "--num_modes", str(job.get("num_modes",1)),
        "--cpu", str(job.get("cpu",1)),
        "--out", str(out_file), "--seed", str(job.get("seed",42))
    ]
    start=time.time()
    rc=1
    try:
        with open(log_file, "w") as lf:
            lf.write(" ".join(cmd)+"\n")
            rc=subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    except Exception as e:
        with open(log_file, "a") as lf:
            lf.write("ERROR: "+str(e)+"\n")
        rc=-1
    runtime = round(time.time()-start,2)
    status = "ok" if rc==0 else "error"
    return {**job, "log_file": str(log_file.resolve()), "return_code": rc, "status": status, "runtime_s": runtime}

def run_job_gpu(job, vina_bin, gpu_id):
    receptor = job["receptor"]
    ligand = job["ligand"]
    output_file = job["output_file"]
    out_dir = Path(output_file).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "log"; log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{Path(ligand).stem}_gpu{gpu_id}.log"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = (
        f"{ENV_SETUP}; {vina_bin} "
        f"--receptor {receptor} "
        f"--ligand {ligand} "
        f"--out {output_file} "
        f"--center_x {job['center_x']} --center_y {job['center_y']} --center_z {job['center_z']} "
        f"--size_x {job['size_x']} --size_y {job['size_y']} --size_z {job['size_z']} "
        f"--thread {job.get('threads',8000)}"
    )
    start = time.time()
    rc = subprocess.call(cmd, shell=True, stdout=open(log_file, "w"),
                         stderr=subprocess.STDOUT, executable="/bin/bash")
    runtime = round(time.time()-start,2)
    status = "ok" if rc==0 else "error"
    return {**job, "gpu_id": gpu_id, "log_file": str(log_file.resolve()), "return_code": rc, "status": status, "runtime_s": runtime}

def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print("CSV not found:", csv_path)
        return

    # detect total CPUs from SLURM or system
    total_cpus = 0
    for k in ("SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_PER_NODE"):
        v = os.environ.get(k)
        if v:
            try:
                total_cpus = int(v)
                break
            except ValueError:
                pass
    if total_cpus == 0:
        total_cpus = multiprocessing.cpu_count()

    cpu_per_job = max(1, args.cpu)
    workers = args.workers if args.workers > 0 else max(1, total_cpus // cpu_per_job)

    shard_idx = args.shard_idx if args.shard_idx >= 0 else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0") or 0)
    shard_total = args.shard_total if args.shard_total > 1 else int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1") or 1)

    jobs = build_jobs_from_boxes(csv_path, cpu_per_job, args.threads, limit=(args.dry_run or 0), test_run=args.test_run)
    if args.dry_run:
        jobs = jobs[: args.dry_run]

    if shard_total > 1:
        jobs = [j for j in jobs if stable_hash_mod(j.get("ligand_name") or j["ligand"], shard_total) == shard_idx]

    if args.skip_existing:
        jobs = [j for j in jobs if not Path(j["output_file"]).exists()]

    if not jobs:
        print("No jobs to run after sharding/skip_existing.")
        return

    vina = resolve_vina_bin(args.vina_bin)

    # Ensure gpu_id key exists in all job dicts (None for CPU mode)
    for j in jobs:
        j.setdefault("gpu_id", None)

    manifest = csv_path.parent / "docking_manifest.csv"
    mf_exists = manifest.exists()
    mf_fh = open(manifest, "a", newline="")
    # Build manifest columns
    fieldnames = list(jobs[0].keys()) + ["log_file", "return_code", "status", "runtime_s"]
    if "gpu_id" not in fieldnames:
        fieldnames.append("gpu_id")
    writer = csv.DictWriter(mf_fh, fieldnames=fieldnames)
    if (not mf_exists) or manifest.stat().st_size == 0:
        writer.writeheader()

    print(f"[master] mode={args.mode} Prepared {len(jobs)} jobs. workers={workers} cpu_per_job={cpu_per_job}")

    try:
        if args.mode == "cpu":
            pool = Pool(processes=workers)
            try:
                for res in pool.imap_unordered(lambda jb: run_job_cpu(jb, vina), jobs):
                    with MANIFEST_LOCK:
                        writer.writerow(res)
                        mf_fh.flush()
            finally:
                pool.close()
                pool.join()
        else:  # GPU mode
            for j in jobs:
                res = run_job_gpu(j, str(Path(args.vina_bin).resolve()), args.gpu_id)
                with MANIFEST_LOCK:
                    writer.writerow(res)
                    mf_fh.flush()
    finally:
        mf_fh.close()

if __name__ == "__main__":
    main()
