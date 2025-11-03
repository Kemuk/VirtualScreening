#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Keep BLAS/OpenMP libraries from oversubscribing CPUs inside each worker
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

# ---------- helpers ----------
def five_number_summary(vals):
    if not vals:
        return ["NA"] * 5
    arr = np.asarray(vals, dtype=float)
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    return [int(np.min(arr)), int(q1), int(med), int(q3), int(np.max(arr))]

def num_rot_bonds_from_smiles(smiles):
    # Import inside worker-safe function
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return int(Descriptors.NumRotatableBonds(mol))

def read_lit_pcba_target(target_dir):
    actives_path = os.path.join(target_dir, "actives.smi")
    inactives_path = os.path.join(target_dir, "inactives.smi")
    actives, inactives = [], []

    if os.path.isfile(actives_path):
        with open(actives_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.strip().split()
                if not parts: continue
                rb = num_rot_bonds_from_smiles(parts[0])
                if rb is not None: actives.append(rb)

    if os.path.isfile(inactives_path):
        with open(inactives_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.strip().split()
                if not parts: continue
                rb = num_rot_bonds_from_smiles(parts[0])
                if rb is not None: inactives.append(rb)

    return actives, inactives

def read_dekois2_target(target_dir):
    comb_path = os.path.join(target_dir, "active_decoys.smi")
    actives, decoys = [], []
    if not os.path.isfile(comb_path):
        return actives, decoys

    with open(comb_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 2: continue
            smi, lig_id = parts[0], parts[1]
            rb = num_rot_bonds_from_smiles(smi)
            if rb is None: continue
            if lig_id.startswith("BDB"):      # active
                actives.append(rb)
            elif lig_id.startswith("ZINC"):   # decoy
                decoys.append(rb)
            # else: ignore
    return actives, decoys

def canonical_dataset_name(name):
    return "LIT-PCBA" if name in ("LIT-PCBA", "LIT_PCBA") else name

def enumerate_targets(roots):
    """Yield tuples (dataset, target, target_dir)."""
    for root in roots:
        if not os.path.isdir(root): continue
        dataset = canonical_dataset_name(os.path.basename(os.path.normpath(root)))
        for entry in sorted(os.listdir(root)):
            target_dir = os.path.join(root, entry)
            if os.path.isdir(target_dir):
                yield (dataset, entry, target_dir)

def process_one_target(task):
    """Worker function. Returns a dict row for the master DataFrame."""
    dataset, target, target_dir = task
    if dataset == "LIT-PCBA":
        a_rbs, d_rbs = read_lit_pcba_target(target_dir)   # d_rbs = inactives
    else:
        a_rbs, d_rbs = read_dekois2_target(target_dir)    # d_rbs = decoys

    all_rbs = a_rbs + d_rbs
    min_all, q1_all, med_all, q3_all, max_all = five_number_summary(all_rbs)
    min_a, q1_a, med_a, q3_a, max_a = five_number_summary(a_rbs)
    min_d, q1_d, med_d, q3_d, max_d = five_number_summary(d_rbs)

    return {
        "Dataset": dataset,
        "Target": target,
        "NumberActives": len(a_rbs),
        "NumberDecoys/Inactives": len(d_rbs),
        "Minimum_All": min_all,
        "FirstQuartile_All": q1_all,
        "Median_All": med_all,
        "ThirdQuartile_All": q3_all,
        "Maximum_All": max_all,
        "Minimum_Actives": min_a,
        "FirstQuartile_Actives": q1_a,
        "Median_Actives": med_a,
        "ThirdQuartile_Actives": q3_a,
        "Maximum_Actives": max_a,
        "Minimum_Decoys/Inactives": min_d,
        "FirstQuartile_Decoys/Inactives": q1_d,
        "Median_Decoys/Inactives": med_d,
        "ThirdQuartile_Decoys/Inactives": q3_d,
        "Maximum_Decoys/Inactives": max_d,
    }

def default_workers():
    # Respect SLURM/PBS if present; otherwise use all local CPUs
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE", "PBS_NP"):
        if var in os.environ:
            try:
                v = int(str(os.environ[var]).split("(")[0].split(",")[0])
                if v > 0: return v
            except Exception:
                pass
    return os.cpu_count() or 1

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Parallel DEKOIS2 / LIT-PCBA summarizer with tqdm.")
    ap.add_argument("--roots", nargs="*", default=["LIT-PCBA", "LIT_PCBA", "DEKOIS2"],
                    help="Dataset roots to scan (default: LIT-PCBA, LIT_PCBA, DEKOIS2 if present).")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help="Number of worker processes (default: auto from SLURM/PBS or cpu_count).")
    args = ap.parse_args()

    tasks = list(enumerate_targets(args.roots))
    if not tasks:
        raise SystemExit("No targets found under the given roots.")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one_target, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing targets", unit="target"):
            rows.append(fut.result())

    df = pd.DataFrame(rows).sort_values(["Dataset", "Target"]).reset_index(drop=True)

    # Save both filtered views from the same DataFrame
    df.to_csv("dataset_summary.csv", index=False)

    lit = df[df["Dataset"] == "LIT-PCBA"].copy()
    lit.rename(columns={
        "NumberDecoys/Inactives": "NumberInactives",
        "Minimum_Decoys/Inactives": "Minimum_Inactives",
        "FirstQuartile_Decoys/Inactives": "FirstQuartile_Inactives",
        "Median_Decoys/Inactives": "Median_Inactives",
        "ThirdQuartile_Decoys/Inactives": "ThirdQuartile_Inactives",
        "Maximum_Decoys/Inactives": "Maximum_Inactives",
    }, inplace=True)
    lit.to_csv("per_target_summary.csv", index=False)

    print(f"[OK] Master DataFrame rows: {len(df)}")
    print(f"[OK] Wrote dataset_summary.csv (all datasets)")
    print(f"[OK] Wrote per_target_summary.csv (LIT-PCBA only)")

if __name__ == "__main__":
    main()
