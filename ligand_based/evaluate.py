#!/usr/bin/env python3
# evaluate.py
import os, sys, math, subprocess
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count
from functools import partial

# optional RDKit scoring helpers (bedroc, enrichment) -- let import fail if unavailable
try:
    from rdkit.ML.Scoring import Scoring as RDScoring
except Exception:
    RDScoring = None

# descriptor sizes
USR_N = 12
USRCAT_N = 60
ES_N = 15

# -------------------------
# small helper: read master CSV and compress to npz
# -------------------------
def safe_array(df, cols, n):
    import numpy as _np
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return _np.full((len(df), n), _np.nan, dtype=float)
    a = df[cols].astype(float).replace([_np.inf, -_np.inf], _np.nan).to_numpy()
    if a.shape[1] != n:
        out = _np.full((len(df), n), _np.nan, dtype=float)
        m = min(n, a.shape[1])
        out[:, :m] = a[:, :m]
        return out
    return a

# --- WORKER FUNCTION ---
def process_chunk(chunk_df, usr_cols, usrcat_cols, es_cols):
    ids_arr = chunk_df.get('id', pd.Series([str(i) for i in range(len(chunk_df))])).astype(str).values
    smiles_arr = chunk_df.get('smiles', pd.Series(['']*len(chunk_df))).astype(str).values
    labels_arr = chunk_df.get('label', pd.Series(['0']*len(chunk_df))).astype(int).values
    targets_arr = chunk_df.get('Protein_ID', pd.Series(['']*len(chunk_df))).astype(str).values
    refs_arr = chunk_df.get('ref_ligand_path', pd.Series(['']*len(chunk_df))).astype(str).values
    usr_arr = safe_array(chunk_df, usr_cols, USR_N)
    usrcat_arr = safe_array(chunk_df, usrcat_cols, USRCAT_N)
    es_arr = safe_array(chunk_df, es_cols, ES_N)
    del chunk_df
    return (ids_arr, smiles_arr, labels_arr, targets_arr, refs_arr, usr_arr, usrcat_arr, es_arr)

# minimal mol2 -> descriptors. Let failures propagate.
def mol2_descriptors_from_file(path):
    m = Chem.MolFromMol2File(path, removeHs=False)
    if m is None:
        txt = open(path, 'r').read()
        m = Chem.MolFromMol2Block(txt, removeHs=False)
    if m is None:
        raise ValueError(f"Failed to parse mol2: {path}")
    if not m.GetNumConformers():
        m = Chem.AddHs(Chem.Mol(m))
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(m, params)
        AllChem.MMFFOptimizeMolecule(m, confId=0)
    usr = list(rdMolDescriptors.GetUSR(m, 0))
    ucat = rdMolDescriptors.GetUSRCAT(m)
    ucat = np.array(ucat, dtype=float)
    if ucat.ndim == 2:
        ucat = ucat[0]
    es = None
    try:
        from oddt import toolkit
        from oddt.shape import electroshape
        mb = Chem.MolToMolBlock(m, confId=0)
        oddt_m = toolkit.readstring('sdf', mb)
        es = list(electroshape(oddt_m))
    except Exception:
        es = [float('nan')] * ES_N
    return np.array(usr, dtype=float), np.array(ucat, dtype=float), np.array(es, dtype=float)

# -------------------------
# metrics
# -------------------------
def compute_point_metrics(labels, scores, higher_is_better=True):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    
    # --- ADDED CHECK ---
    # Handle NaN scores, which can come from failed descriptor calcs
    valid_mask = ~np.isnan(scores)
    if np.sum(valid_mask) < 2: # Not enough data to score
        return {"ROC-AUC": np.nan, "PR-AUC": np.nan, "BEDROC(20)": np.nan,
                "NEF1%": np.nan, "NEF5%": np.nan, "NEF10%": np.nan}
    labels = labels[valid_mask]
    scores = scores[valid_mask]
    # --- END CHECK ---

    if np.unique(labels).size < 2:
        return {"ROC-AUC": np.nan, "PR-AUC": np.nan, "BEDROC(20)": np.nan,
                "NEF1%": np.nan, "NEF5%": np.nan, "NEF10%": np.nan}
    try:
        fpr,tpr,_ = roc_curve(labels, scores)
        roc_auc = auc(fpr,tpr)
    except Exception:
        roc_auc = np.nan
    try:
        prec,rec,_ = precision_recall_curve(labels, scores)
        pr_auc = auc(rec,prec)
    except Exception:
        pr_auc = np.nan
    if RDScoring is not None:
        # RDKit scoring tools need scores to be higher-is-better.
        # If ours is lower-is-better (like distance), we must invert it.
        if not higher_is_better:
            scores = -scores
            
        arr = [[float(s), bool(y)] for s,y in zip(scores, labels)]
        try:
            bed = float(RDScoring.CalcBEDROC(arr, 1, 20.0))
        except Exception:
            bed = np.nan
        try:
            efs = RDScoring.CalcEnrichment(arr, 1, [0.01,0.05,0.10])
            ef1 = float(efs[0])
            ef5 = float(efs[1])
            ef10 = float(efs[2])
            n = len(labels); npos = int(labels.sum())
            def nef_from_ef(ef, frac):
                top_n = max(1, math.ceil(n*frac))
                base = npos / n if n>0 else 0
                if base <= 0: return np.nan
                ef_max = (min(npos, top_n)/top_n) / base
                return ef/ef_max if ef_max>0 else np.nan
            nef1 = nef_from_ef(ef1, 0.01)
            nef5 = nef_from_ef(ef5, 0.05)
            nef10 = nef_from_ef(ef10, 0.10)
        except Exception:
            bed, nef1, nef5, nef10 = np.nan, np.nan, np.nan, np.nan
    else:
        bed, nef1, nef5, nef10 = np.nan, np.nan, np.nan, np.nan

    return {"ROC-AUC": float(roc_auc), "PR-AUC": float(pr_auc), "BEDROC(20)": float(bed),
            "NEF1%": float(nef1), "NEF5%": float(nef5), "NEF10%": float(nef10)}


# --- NEW METRIC FUNCTION ---
def euclidean_dist_matrix(vec_ref, mat):
    """Calculates Euclidean distance between a 1D ref vector and rows of a 2D matrix."""
    if vec_ref is None or np.isnan(vec_ref).any():
        return np.full((mat.shape[0],), np.nan, dtype=float)
    # Use broadcasting to calculate difference, then norm along the feature axis
    return np.linalg.norm(mat - vec_ref, axis=1)
# --- END NEW METRIC FUNCTION ---


# -------------------------
# main
# -------------------------
def main():
    base = "." 
    outdir = os.path.join(base, "output")
    master_csv = os.path.join(outdir, "LIT_PCBA_predictions.csv")
    npz_path = os.path.join(outdir, "descriptors.npz")

    if os.path.exists(npz_path):
        print(f"Loading cached descriptors from {npz_path}")
        try:
            data = np.load(npz_path, allow_pickle=True)
            ids = data['ids']
            smiles = data['smiles']
            labels = data['labels']
            targets = data['targets']
            refs = data['refs']
            usr = data['usr']
            usrcat = data['usrcat']
            es = data['es']
        except Exception as e:
            print(f"Failed to load {npz_path}, rebuilding... Error: {e}", file=sys.stderr)
            try: os.remove(npz_path)
            except OSError: pass
            return main()
            
    else:
        print(f"Cache not found. Building from {master_csv} using parallel chunking...")
        if not os.path.exists(master_csv):
            raise SystemExit(f"Master CSV not found: {master_csv}")

        core_cols = ['id', 'smiles', 'label', 'Protein_ID', 'ref_ligand_path']
        USR_cols = [f"USR_{i}" for i in range(USR_N)]
        USRCAT_cols = [f"USRCAT_{i}" for i in range(USRCAT_N)]
        ES_cols = [f"ES_{i}" for i in range(ES_N)]
        all_cols = core_cols + USR_cols + USRCAT_cols + ES_cols
        
        chunksize = 100_000
        
        try:
            chunk_iter = pd.read_csv(master_csv, dtype=str, keep_default_na=False,
                                     usecols=lambda c: c in all_cols, chunksize=chunksize)
        except ValueError:
            print(f"Warning: 'usecols' failed. Loading full CSV in chunks...", file=sys.stderr)
            chunk_iter = pd.read_csv(master_csv, dtype=str, keep_default_na=False, chunksize=chunksize)

        n_workers = cpu_count()
        print(f"Starting parallel processing with {n_workers} workers...")
        
        worker_func = partial(process_chunk, usr_cols=USR_cols, 
                              usrcat_cols=USRCAT_cols, es_cols=ES_cols)

        results = process_map(worker_func, chunk_iter, 
                              max_workers=n_workers, chunksize=1, desc="Processing CSV chunks")
        
        print("All chunks processed. Concatenating arrays...")
        
        all_results = list(zip(*results))
        ids = np.concatenate(all_results[0])
        smiles = np.concatenate(all_results[1])
        labels = np.concatenate(all_results[2])
        targets = np.concatenate(all_results[3])
        refs = np.concatenate(all_results[4])
        usr = np.concatenate(all_results[5], axis=0)
        usrcat = np.concatenate(all_results[6], axis=0)
        es = np.concatenate(all_results[7], axis=0)
        
        del results, all_results
        
        os.makedirs(outdir, exist_ok=True)
        np.savez_compressed(npz_path, ids=ids, smiles=smiles, labels=labels, targets=targets, refs=refs, usr=usr, usrcat=usrcat, es=es)
        print("Wrote cache to", npz_path)

    unique_targets = np.unique(targets)
    ref_desc = {}
    print(f"Precomputing reference descriptors for {len(unique_targets)} targets...")
    
    # --- ADDED DEBUGGING ---
    debug_count = 0
    # --- END DEBUGGING ---

    for t in tqdm(unique_targets, desc="Precomputing refs"):
        mask = (targets == t)
        ref_paths = refs[mask]
        ref_path = next((r for r in ref_paths if r), "")
        if not ref_path:
            ref_desc[t] = (None, None, None)
            continue
        try:
            u_r, uc_r, es_r = mol2_descriptors_from_file(ref_path)
            ref_desc[t] = (u_r, uc_r, es_r)
            
            # --- ADDED DEBUGGING ---
            if debug_count < 3: # Print info for the first 3 valid targets
                print(f"\nDEBUG: Target {t}")
                print(f"  Ref path: {ref_path}")
                print(f"  USR mean: {np.mean(u_r)}, USRCAT mean: {np.mean(uc_r)}, ES mean: {np.mean(es_r)}")
                print(f"  USR norm: {np.linalg.norm(u_r)}, USRCAT norm: {np.linalg.norm(uc_r)}, ES norm: {np.linalg.norm(es_r)}")
                debug_count += 1
            # --- END DEBUGGING ---

        except Exception as e:
            print(f"Warning: Failed to compute descriptors for target {t} (path: {ref_path}). Error: {e}", file=sys.stderr)
            ref_desc[t] = (None, None, None)

    print("Building scores per compound...")
    rows = []
    for i in tqdm(range(len(ids)), desc="Building scores"):
        _id = ids[i]
        prot = targets[i]
        uref, ucr, esr = ref_desc.get(prot, (None,None,None))
        row = {
            "id": _id,
            "smiles": smiles[i],
            "label": int(labels[i]),
            "Protein_ID": prot,
            # --- MODIFIED: Call new function ---
            "USR_score": float(euclidean_dist_matrix(uref, usr[i:i+1])[0]),
            "USRCAT_score": float(euclidean_dist_matrix(ucr, usrcat[i:i+1])[0]),
            "Electroshape_score": float(euclidean_dist_matrix(esr, es[i:i+1])[0]),
        }
        rows.append(row)
    scores_df = pd.DataFrame(rows)
    scores_csv = os.path.join(outdir, "scores.csv")
    scores_df.to_csv(scores_csv, index=False)
    print("Wrote", scores_csv)

    print("Calculating per-target metrics...")
    methods = {
        # --- MODIFIED: higher_is_better is now False ---
        "USR": ("USR_score", False),
        "USRCAT": ("USRCAT_score", False),
        "Electroshape": ("Electroshape_score", False)
    }
    per_rows = []
    for t in tqdm(unique_targets, desc="Per-target metrics"):
        msk = (scores_df["Protein_ID"] == t)
        sub = scores_df.loc[msk]
        lab = sub["label"].astype(int).values
        row = {"Protein_ID": t, "N Actives": int(lab.sum()), "N Compounds": int(msk.sum())}
        for mname, (col, hib) in methods.items():
            vals = sub[col].astype(float).values
            metrics = compute_point_metrics(lab, vals, higher_is_better=hib) # `hib` is now False
            for k,v in metrics.items():
                row[f"{mname} {k}"] = (f"{v:.3f}" if (v==v) else "nan")
                row[f"{mname} {k}_value"] = v
        per_rows.append(row)
    per_df = pd.DataFrame(per_rows)
    per_df.to_csv(os.path.join(outdir, "per_target_metrics.csv"), index=False)
    print("Wrote per_target_metrics.csv")

    print("Calculating global metrics...")
    global_rows = []
    labels_all = scores_df["label"].astype(int).values
    for mname, (col, hib) in methods.items():
        vals = scores_df[col].astype(float).values
        metrics = compute_point_metrics(labels_all, vals, higher_is_better=hib) # `hib` is now False
        global_rows.append({"Method": mname, **metrics})
    pd.DataFrame(global_rows).set_index("Method").to_csv(os.path.join(outdir, "global_metrics.csv"))
    print("Wrote global_metrics.csv")

    print("Running plots.py...")
    try:
        subprocess.run(["python3", os.path.join(base, "plots.py"), base], check=True)
    except subprocess.CalledProcessError as e:
        print("plots.py failed:", e, file=sys.stderr)
    except FileNotFoundError:
        print("plots.py not found. Skipping plotting.", file=sys.stderr)

if __name__ == "__main__":
    main()