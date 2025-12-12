#!/usr/bin/env python3
"""
Parallel builder for aev_plig.csv from vina_boxes.csv.
- Task prep in threads (fast globbing).
- RDKit work in processes.
- array_id assigned in worker via random.randint.
- Streams outputs to disk.
"""
import argparse, csv, gzip, os, sys, time, random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import thread_map as tmap
from tqdm.auto import tqdm

from rdkit import rdBase
import warnings
rdBase.DisableLog('rdApp.warning')
warnings.filterwarnings("ignore", category=UserWarning)


MAIN_COLS = [
    "unique_id", "Protein_ID", "sdf_file", "protein_pdb",
    "MW", "LogP", "HBD", "HBA", "DockingScore", "pK", "is_active"
]
REQUIRED = ["target","receptor_pdbqt","docked_sdf_actives","docked_sdf_inactives","log_actives","log_inactives"]

# --- RNG + config shared to workers ---
SHARD_TOTAL = 1
def _init_worker(shard_total: int):
    global SHARD_TOTAL
    SHARD_TOTAL = max(1, int(shard_total))
    # unique seed per process
    random.seed(os.getpid() ^ (time.time_ns() & 0xFFFFFFFF))

# --- RDKit helpers ---
def compute_rdkit_props(sdf_file: str) -> Optional[Tuple[float, float, int, int]]:
    try:
        suppl = Chem.SDMolSupplier(sdf_file, sanitize=False)
        mol = next((m for m in suppl if m is not None), None)
        if not mol: return None
        return (Descriptors.MolWt(mol), Crippen.MolLogP(mol),
                Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol))
    except Exception:
        return None

def parse_vina_log(log_file: str) -> Optional[float]:
    try:
        with open(log_file, "r") as fh:
            for line in fh:
                ls = line.strip()
                if ls.startswith("1 ") or ls == "1":
                    parts = ls.split()
                    if len(parts) >= 2: return float(parts[1])
        return None
    except Exception:
        return None

def docking_to_pK(dg: float, temp: float = 298.0) -> float:
    R = 0.001987
    return -dg / (2.303 * R * temp)

def find_protein_pdb_from_receptor(receptor_pdbqt: Optional[str]) -> Optional[str]:
    try:
        if not receptor_pdbqt: return None
        p = Path(receptor_pdbqt)
        if not p.exists(): return None
        cand = p.with_suffix(".pdb")
        if cand.exists(): return str(cand.resolve())
        for q in p.parent.glob("*.pdb"): return str(q.resolve())
        return None
    except Exception:
        return None

# --- worker ---
def process_ligand(task: Tuple) -> Optional[Dict]:
    target, sdf_path, log_dir, is_active, receptor_pdbqt = task
    lig_id = Path(sdf_path).stem
    props = compute_rdkit_props(str(sdf_path))
    if not props: return None
    mw, logp, hbd, hba = props
    log_file = Path(log_dir) / f"{lig_id}.log"
    score = parse_vina_log(str(log_file)) if log_file.exists() else None
    pk = docking_to_pK(score) if score is not None else None
    protein_pdb = find_protein_pdb_from_receptor(receptor_pdbqt)
    return {
        "unique_id": lig_id,
        "Protein_ID": target,
        "sdf_file": str(Path(sdf_path).resolve()),
        "protein_pdb": protein_pdb,
        "MW": mw, "LogP": logp, "HBD": hbd, "HBA": hba,
        "DockingScore": score, "pK": pk, "is_active": is_active,
        "array_id": random.randint(0, SHARD_TOTAL - 1),
    }

# --- parallel task prep ---
def collect_tasks(boxes_csv: Path, workers: int = 0, test_run: bool = False) -> List[Tuple]:
    df = pd.read_csv(boxes_csv)
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss: raise SystemExit(f"Missing columns: {miss}")
    recs = df.to_dict(orient="records")
    workers = workers or min(64, max(1, cpu_count()))

    act_dirs = [Path(r["docked_sdf_actives"]) for r in recs if isinstance(r.get("docked_sdf_actives"), str)]
    inact_dirs = [Path(r["docked_sdf_inactives"]) for r in recs if isinstance(r.get("docked_sdf_inactives"), str)]
    uniq_dirs = list(set(act_dirs) | set(inact_dirs))

    def glob_dir(d: Path):
        if not d.is_dir(): return (d, [])
        return (d, sorted(d.glob("*.sdf")) + sorted(d.glob("*.SDF")))
    glob_map = dict(tmap(glob_dir, uniq_dirs, max_workers=workers, desc="Globbing SDF dirs"))

    def build(rec: dict):
        target = str(rec["target"])
        receptor = str(rec["receptor_pdbqt"]) if pd.notna(rec.get("receptor_pdbqt")) else None
        out = []
        pairs = [
            (Path(str(rec.get("docked_sdf_actives"))), Path(str(rec.get("log_actives"))), 1),
            (Path(str(rec.get("docked_sdf_inactives"))), Path(str(rec.get("log_inactives"))), 0),
        ]
        for sdf_dir, log_dir, is_act in pairs:
            for f in glob_map.get(sdf_dir, []):
                out.append((target, f, log_dir, is_act, receptor))
                if test_run: return out
        return out

    per = tmap(build, recs, max_workers=workers, desc="Building tasks")
    return [t for sub in per for t in sub]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="vina_boxes.csv")
    ap.add_argument("--out", default="aev_plig.csv")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--prepare_workers", type=int, default=0)
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--shard_total", type=int, default=100)
    ap.add_argument("--test_run", action="store_true")
    args = ap.parse_args()

    tasks = collect_tasks(Path(args.csv), workers=args.prepare_workers, test_run=args.test_run)
    print(f"Prepared tasks: {len(tasks)}")
    if not tasks: sys.exit(0)

    workers = args.workers or max(1, cpu_count())
    total = len(tasks)
    chunksize = max(1, total // (workers * 8))

    out_path = Path(args.out)
    out_file = str(out_path) + (".gz" if args.compress else "")
    ds_file = ("dataset.csv.gz" if args.compress else "dataset.csv")

    open_main = gzip.open if args.compress else open
    open_ds = gzip.open if args.compress else open
    header = MAIN_COLS + ["array_id"]

    with open_main(out_file, "wt", newline="") as fh_main, open_ds(ds_file, "wt", newline="") as fh_ds:
        w_main = csv.DictWriter(fh_main, fieldnames=header); w_main.writeheader()
        w_ds = csv.DictWriter(fh_ds, fieldnames=["unique_id","sdf_file","pdb_file"]); w_ds.writeheader()
        with Pool(processes=workers, initializer=_init_worker, initargs=(args.shard_total,)) as pool:
            for res in tqdm(pool.imap_unordered(process_ligand, tasks, chunksize=chunksize),
                            total=total, desc="Processing ligands", unit="lig"):
                if not res: continue
                w_main.writerow({k: res.get(k) for k in header})
                w_ds.writerow({"unique_id": res["unique_id"], "sdf_file": res["sdf_file"], "pdb_file": res["protein_pdb"]})

    print(f"Wrote → {out_file}")
    print(f"Wrote → {ds_file}")

if __name__ == "__main__":
    main()
