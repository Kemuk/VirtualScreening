#!/usr/bin/env python3
"""
Simple parallel builder for aev_plig.csv from vina_boxes.csv.

Behavior:
- Reads 'vina_boxes.csv' to find SDF directories ('docked_sdf_...')
  and Vina PDBQT directories ('docked_vina_...').
- For each SDF, it looks for the corresponding PDBQT file:
  {vina_dir}/{lig_id}_docked.pdbqt
- Parses the first Vina REMARK affinity and converts to pK.
- Minimal, deterministic, and fast.
"""
import argparse
import csv
import os
import random
import re
import sys
import time
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
# MODIFIED: Updated to reflect the columns we actually use.
REQUIRED = [
    "target", "receptor_pdbqt", 
    "docked_sdf_actives", "docked_sdf_inactives",
    "docked_vina_actives", "docked_vina_inactives"
]

SHARD_TOTAL = 1
def _init_worker(shard_total: int):
    global SHARD_TOTAL
    SHARD_TOTAL = max(1, int(shard_total))
    random.seed(os.getpid() ^ (time.time_ns() & 0xFFFFFFFF))

# regex flexible to match common Vina remark lines
_REMARK_VINA_RE = re.compile(r"REMARK.*VINA(?:\s+RESULT)?[:\s]+\s*(-?\d+\.\d+)", re.IGNORECASE)

def compute_rdkit_props(sdf_file: str) -> Optional[Tuple[float, float, int, int]]:
    try:
        suppl = Chem.SDMolSupplier(sdf_file, sanitize=False)
        mol = next((m for m in suppl if m is not None), None)
        if not mol:
            return None
        return (Descriptors.MolWt(mol), Crippen.MolLogP(mol),
                Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol))
    except Exception:
        return None

def parse_affinity_from_pdbqt(pdbqt_path: Path) -> Optional[float]:
    """Open pdbqt and return first Vina remark affinity found."""
    if not pdbqt_path or not pdbqt_path.exists():
        return None
    try:
        with open(pdbqt_path, "r") as fh:
            for line in fh:
                m = _REMARK_VINA_RE.search(line)
                if m:
                    try:
                        return float(m.group(1))
                    except Exception:
                        return None
    except Exception:
        return None
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
        for q in p.parent.glob("*.pdb"):
            return str(q.resolve())
        return None
    except Exception:
        return None

def process_ligand(task: Tuple) -> Optional[Dict]:
    """
    MODIFIED
    task: (target, sdf_path, vina_dir, is_active, receptor_pdbqt)
    """
    # MODIFIED: 'vina_dir' is now the path from 'docked_vina_...'
    target, sdf_path, vina_dir, is_active, receptor_pdbqt = task
    
    sdf_path = Path(sdf_path)
    lig_id = sdf_path.stem
    props = compute_rdkit_props(str(sdf_path))
    if not props:
        return None
    mw, logp, hbd, hba = props

    # --- MODIFIED SECTION ---
    # This is the correct path, using the directory from the CSV
    # e.g., /.../docked_vina/actives/ligand1_docked.pdbqt
    pdbqt_path = vina_dir / f"{lig_id}_docked.pdbqt"

    # Try parsing that single, correct file.
    score = parse_affinity_from_pdbqt(pdbqt_path)
    # --- END MODIFIED SECTION ---

    pk = docking_to_pK(score) if score is not None else None
    protein_pdb = find_protein_pdb_from_receptor(receptor_pdbqt)
    return {
        "unique_id": lig_id,
        "Protein_ID": target,
        "sdf_file": str(sdf_path.resolve()),
        "protein_pdb": protein_pdb,
        "MW": mw, "LogP": logp, "HBD": hbd, "HBA": hba,
        "DockingScore": score, "pK": pk, "is_active": is_active,
    }

def collect_tasks(boxes_csv: Path, workers: int = 0, test_run: bool = False) -> List[Tuple]:
    df = pd.read_csv(boxes_csv)
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise SystemExit(f"Missing columns: {miss}")
    recs = df.to_dict(orient="records")
    workers = workers or min(64, max(1, cpu_count()))

    act_dirs = [Path(r["docked_sdf_actives"]) for r in recs if isinstance(r.get("docked_sdf_actives"), str) and pd.notna(r.get("docked_sdf_actives"))]
    inact_dirs = [Path(r["docked_sdf_inactives"]) for r in recs if isinstance(r.get("docked_sdf_inactives"), str) and pd.notna(r.get("docked_sdf_inactives"))]
    uniq_dirs = list({d for d in (act_dirs + inact_dirs) if d is not None})

    def glob_dir(d: Path):
        if not d.is_dir():
            return (d, [])
        files = sorted(d.glob("*.sdf")) + sorted(d.glob("*.SDF"))
        return (d, files)

    glob_map = dict(tmap(glob_dir, uniq_dirs, max_workers=workers, desc="Globbing SDF dirs"))

    def build(rec: dict):
        target = str(rec.get("target"))
        receptor = str(rec.get("receptor_pdbqt")) if pd.notna(rec.get("receptor_pdbqt")) else None
        out = []
        
        # MODIFIED: Now includes 'vina_dir' from the CSV
        pairs = [
            (Path(str(rec.get("docked_sdf_actives"))) if pd.notna(rec.get("docked_sdf_actives")) else None,
             Path(str(rec.get("docked_vina_actives"))) if pd.notna(rec.get("docked_vina_actives")) else None,
             1),
            (Path(str(rec.get("docked_sdf_inactives"))) if pd.notna(rec.get("docked_sdf_inactives")) else None,
             Path(str(rec.get("docked_vina_inactives"))) if pd.notna(rec.get("docked_vina_inactives")) else None,
             0),
        ]
        
        # MODIFIED: 'vina_dir' is now passed instead of 'log_dir'
        for sdf_dir, vina_dir, is_act in pairs:
            if sdf_dir is None or vina_dir is None: # Check both are valid
                continue
            for f in glob_map.get(sdf_dir, []):
                out.append((target, f, vina_dir, is_act, receptor))
                if test_run:
                    return out
        return out

    per = tmap(build, recs, max_workers=workers, desc="Building tasks")
    return [t for sub in per for t in sub]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="LIT_PCBA/vina_boxes.csv")
    ap.add_argument("--out", default="LIT_PCBA/aev_plig.csv")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--prepare_workers", type=int, default=0)
    ap.add_argument("--shard_total", type=int, default=100)
    ap.add_argument("--test_run", action="store_true")
    args = ap.parse_args()

    tasks = collect_tasks(Path(args.csv), workers=args.prepare_workers, test_run=args.test_run)
    print(f"Prepared tasks: {len(tasks)}")
    if not tasks:
        sys.exit(0)

    workers = args.workers or max(1, cpu_count())
    total = len(tasks)
    chunksize = max(1, total // (workers * 8))

    out_path = Path(args.out)
    out_file = str(out_path)
    ds_file = "dataset.csv"

    with open(out_file, "wt", newline="") as fh_main, open(ds_file, "wt", newline="") as fh_ds:
        w_main = csv.DictWriter(fh_main, fieldnames=header)
        w_main.writeheader()
        w_ds = csv.DictWriter(fh_ds, fieldnames=["unique_id","sdf_file","pdb_file"])
        w_ds.writeheader()
        with Pool(processes=workers, initializer=_init_worker, initargs=(args.shard_total,)) as pool:
            for res in tqdm(pool.imap_unordered(process_ligand, tasks, chunksize=chunksize),
                            total=total, desc="Processing ligands", unit="lig"):
                if not res:
                    continue
                w_main.writerow({k: res.get(k) for k in header})
                w_ds.writerow({"unique_id": res["unique_id"], "sdf_file": res["sdf_file"], "pdb_file": res["protein_pdb"]})

    print(f"Wrote → {out_file}")
    print(f"Wrote → {ds_file}")

if __name__ == "__main__":
    main()