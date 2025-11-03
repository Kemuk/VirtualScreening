#!/usr/bin/env python3
"""
dock.py

- tqdm is required. Uses tqdm.contrib.concurrent.thread_map for all parallel stages.
- --prepare_master : build master manifest (multithreaded). Rows with Done? == "Yes" get empty task.
  Only undone rows receive contiguous task ids.
- runtime : read master manifest and run rows where task == this task id and Done? == No.
"""
from pathlib import Path
import argparse
import csv
import os
import subprocess
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import random
from tqdm.contrib.concurrent import thread_map as tmap
from tqdm import tqdm

# deterministic seeds
os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)
random.seed(42)

# require tqdm concurrent helper
try:
    # already imported above for convenience
    pass
except Exception as e:
    raise ImportError("tqdm.contrib.concurrent.thread_map is required. Install recent tqdm.") from e

REQUIRED_COLS = [
    "target","receptor_pdbqt","actives_dir","inactives_dir",
    "docked_vina_actives","docked_vina_inactives",
    "center_x","center_y","center_z","size_x","size_y","size_z",
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="LIT_PCBA/vina_boxes.csv")
    p.add_argument("--prepare_master", action="store_true")
    p.add_argument("--master_manifest", default="LIT_PCBA/master_manifest.csv")
    p.add_argument("--shard_total", type=int, default=100)
    p.add_argument("--task_id", type=int, default=-1)
    p.add_argument("--mode", choices=("cpu","gpu"), default="gpu")
    p.add_argument("--vina_bin", default="vina-gpu-dev/QuickVina2-GPU-2-1")
    p.add_argument("--threads", type=int, default=8000)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--test_run", action="store_true")
    p.add_argument("--prepare_workers", type=int, default=0)
    p.add_argument("--runtime_workers", type=int, default=0)
    p.add_argument("--allow_gpu_parallel", action="store_true")
    return p.parse_args()

def _read_numpy_csv(csv_path: Path):
    a = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if a.shape == (): a = np.array([a], dtype=a.dtype)
    return a

def collect_ligand_entries(boxes_csv: Path, test_run: bool = False, workers: int = 0):
    """
    Parallel per-box scan using thread_map (tqdm concurrent). Preserves box order.
    Returns list of (target,receptor,ligpath,outpath,cx,cy,cz,sx,sy,sz).
    """
    import csv as _csv
    with open(boxes_csv, newline="") as f:
        rdr = _csv.reader(f); header = next(rdr)
    missing = [c for c in REQUIRED_COLS if c not in header]
    if missing:
        raise SystemExit(f"Missing columns in boxes CSV: {missing}")
    arr = _read_numpy_csv(boxes_csv)
    cx = arr["center_x"].astype(float); cy = arr["center_y"].astype(float); cz = arr["center_z"].astype(float)
    sx = arr["size_x"].astype(float); sy = arr["size_y"].astype(float); sz = arr["size_z"].astype(float)
    targets = arr["target"]
    receptors = [str(Path(p).resolve()) for p in arr["receptor_pdbqt"]]
    act_in = [Path(p) for p in arr["actives_dir"]]
    inact_in = [Path(p) for p in arr["inactives_dir"]]
    act_out = [Path(p) for p in arr["docked_vina_actives"]]
    inact_out = [Path(p) for p in arr["docked_vina_inactives"]]

    n_boxes = arr.shape[0]
    if workers <= 0:
        workers = min(32, max(1, multiprocessing.cpu_count()))

    def process_box(i):
        local = []
        ligs = sorted(act_in[i].glob("*.pdbqt")) if act_in[i].exists() else []
        for lig in ligs:
            ln = lig.stem
            out = act_out[i] / f"{ln}_docked.pdbqt"
            local.append((targets[i], receptors[i], lig.resolve(), out.resolve(),
                          cx[i], cy[i], cz[i], sx[i], sy[i], sz[i]))
            if test_run:
                break
        ligs = sorted(inact_in[i].glob("*.pdbqt")) if inact_in[i].exists() else []
        for lig in ligs:
            ln = lig.stem
            out = inact_out[i] / f"{ln}_docked.pdbqt"
            local.append((targets[i], receptors[i], lig.resolve(), out.resolve(),
                          cx[i], cy[i], cz[i], sx[i], sy[i], sz[i]))
            if test_run:
                break
        return local

    per_box_lists = tmap(process_box, list(range(n_boxes)), max_workers=workers, desc="Collecting boxes")
    entries = []
    for lst in per_box_lists:
        entries.extend(lst)
    return entries

def _make_job_from_entry(entry):
    target, receptor, ligpath, outpath, cx, cy, cz, sx, sy, sz = entry
    return {
        "target": str(target),
        "receptor": str(receptor),
        "ligand": str(ligpath),
        "ligand_name": Path(ligpath).stem,
        "output_file": str(outpath),
        "center_x": float(cx), "center_y": float(cy), "center_z": float(cz),
        "size_x": float(sx), "size_y": float(sy), "size_z": float(sz),
        "cpu": 1, "threads": 8000,
    }

def prepare_master(boxes_csv: Path, master_manifest: Path, shard_total: int, test_run: bool,
                   prepare_workers: int):
    entries = collect_ligand_entries(boxes_csv, test_run=test_run, workers=prepare_workers or 0)
    n = len(entries)
    if n == 0:
        fieldnames = ["task","Done?","target","receptor","ligand","ligand_name","output_file",
                      "center_x","center_y","center_z","size_x","size_y","size_z","cpu","threads"]
        master_manifest.parent.mkdir(parents=True, exist_ok=True)
        tmp = master_manifest.with_suffix(".tmp")
        with open(tmp, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
        tmp.replace(master_manifest)
        print(f"Wrote empty master manifest: {master_manifest}")
        return

    workers = prepare_workers or min(32, max(1, multiprocessing.cpu_count()))

    # build job dicts in parallel with tqdm thread_map (ordered)
    jobs = tmap(_make_job_from_entry, entries, max_workers=workers, desc="Building jobs")

    # check outputs in parallel with tqdm thread_map
    def exists_of(job): return Path(job["output_file"]).exists()
    done_flags = tmap(exists_of, jobs, max_workers=workers, desc="Checking outputs")

    # compose rows and collect undone indices
    rows = []
    undone_indices = []
    for i, job in enumerate(jobs):
        done = "Yes" if done_flags[i] else "No"
        row = {"task": "", "Done?": done}
        row.update(job)
        rows.append(row)
        if done == "No":
            undone_indices.append(i)

    # assign contiguous task ids across undone indices
    m = len(undone_indices)
    base = m // shard_total if shard_total > 0 else 0
    rem = m % shard_total if shard_total > 0 else 0
    blocks = []
    start = 0
    for t in range(shard_total):
        size = base + (1 if t < rem else 0)
        blocks.append((start, start+size))
        start += size

    for assign_idx in range(m):
        global_idx = undone_indices[assign_idx]
        task = 0
        for t,(s,e) in enumerate(blocks):
            if s <= assign_idx < e:
                task = t
                break
        rows[global_idx]["task"] = str(task)

    # atomic write master manifest
    fieldnames = list(rows[0].keys())
    master_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = master_manifest.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k,"") for k in fieldnames})
    tmp.replace(master_manifest)
    todo_count = sum(1 for r in rows if r["Done?"] == "No")
    print(f"Wrote master manifest: {master_manifest} total={n} todo={todo_count}")

def read_master(master_manifest: Path):
    rows = []
    with open(master_manifest, newline="") as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            for k in ("center_x","center_y","center_z","size_x","size_y","size_z"):
                if k in r and r[k] not in (None,""):
                    r[k] = float(r[k])
            rows.append(r)
    return rows

def run_job_cmd(job, vina_bin):
    outp = job.get("output_file","")
    ligand = job.get("ligand","")
    log = Path(outp).parent / "log" / f"{Path(ligand).stem}_A{os.getenv('SLURM_ARRAY_TASK_ID','NA')}.log"
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        vina_bin,
        "--receptor", job["receptor"],
        "--ligand", job["ligand"],
        "--out", outp,
        "--center_x", str(job["center_x"]), "--center_y", str(job["center_y"]), "--center_z", str(job["center_z"]),
        "--size_x", str(job["size_x"]), "--size_y", str(job["size_y"]), "--size_z", str(job["size_z"]),
        "--seed", "42",
    ]
    if job.get("threads"):
        cmd += ["--thread", str(int(job["threads"]))]
    start = time.time()
    try:
        # capture text directly to avoid manual bytes->str decoding issues
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, text=True)
        txt = proc.stdout or ""
        rc = proc.returncode
    except Exception as e:
        txt = f"EXC: {e}\n"; rc = -1
    tmp_log = log.with_suffix(".tmp")
    # write with explicit utf-8 encoding
    with open(tmp_log, "w", encoding="utf-8", errors="replace") as fh:
        fh.write(" ".join(cmd) + "\n\n")
        fh.write(txt)
    tmp_log.replace(log)
    return rc, round(time.time()-start,2), str(log.resolve())

def runtime_execute(myrows, vina_bin, runtime_workers):
    """
    Parallel runtime execution with progress bar that shows the current ligand and receptor in desc.
    Returns results in same order as myrows: list of (job, rc, runtime_sec, logpath).
    """
    def wrapper(job):
        if Path(job.get("output_file","")).exists():
            return (job, 0, 0.0, "")
        rc, rt, logp = run_job_cmd(job, vina_bin)
        return (job, rc, rt, logp)

    max_workers = runtime_workers or 1
    results_map = {}
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, job in enumerate(myrows):
            fut = ex.submit(wrapper, job)
            futures_map[fut] = i

        with tqdm(total=len(myrows), desc="Running jobs") as pbar:
            for fut in as_completed(futures_map):
                idx = futures_map[fut]
                job = myrows[idx]
                try:
                    res = fut.result()
                except Exception as e:
                    res = (job, -1, 0.0, f"EXC:{e}")
                results_map[idx] = res
                # update progress and show current ligand and receptor names
                ligand_name = Path(job.get("ligand","")).name
                receptor_name = job.get("target", Path(job.get("receptor","")).stem)
                pbar.set_description(f"Running {ligand_name} on {receptor_name}")
                pbar.update(1)

    results = [results_map[i] for i in range(len(myrows))]
    return results

def main():
    args = parse_args()
    boxes = Path(args.csv)
    master = Path(args.master_manifest)
    if args.prepare_master:
        if not boxes.exists():
            raise SystemExit(f"Boxes CSV missing: {boxes}")
        prepare_master(boxes, master, shard_total=args.shard_total, test_run=args.test_run,
                       prepare_workers=args.prepare_workers)
        return

    if not master.exists():
        raise SystemExit("Master manifest missing. Run --prepare_master first.")

    task_id = int(args.task_id if args.task_id >= 0 else os.environ.get("SLURM_ARRAY_TASK_ID", "0") or 0)
    rows = read_master(master)
    myrows = [r for r in rows if r.get("task","") == str(task_id) and r.get("Done?","No") == "No"]
    if not myrows:
        print(f"No todo rows for task {task_id}."); return

    # determine runtime workers
    if args.runtime_workers and args.runtime_workers > 0:
        runtime_workers = args.runtime_workers
    else:
        if args.mode == "gpu":
            runtime_workers = 1 if not args.allow_gpu_parallel else max(1, multiprocessing.cpu_count())
        else:
            runtime_workers = max(1, multiprocessing.cpu_count())

    # set GPU mapping for GPU mode
    if args.mode == "gpu":
        gpu_map = os.environ.get("CUDA_VISIBLE_DEVICES")
        if gpu_map:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_map
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    print(f"[task {task_id}] executing {len(myrows)} jobs with {runtime_workers} thread(s)")
    results = runtime_execute(myrows, args.vina_bin, runtime_workers)

    ok = sum(1 for (_, rc, _, _) in results if rc == 0)
    err = sum(1 for (_, rc, _, _) in results if rc != 0)
    print(f"Completed. OK={ok} ERR={err}")

if __name__ == "__main__":
    main()
