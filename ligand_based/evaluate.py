#!/usr/bin/env python3
# evaluate.py
import os, sys, math, subprocess
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.stats import mannwhitneyu
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count
from functools import partial

try:
    from rdkit.ML.Scoring import Scoring as RDScoring
except Exception:
    RDScoring = None

USR_N = 12
USRCAT_N = 60
ES_N = 15

def safe_array(df, cols, n):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return np.full((len(df), n), np.nan, dtype=float)
    a = df[cols].astype(float).replace([np.inf, -np.inf], np.nan).to_numpy()
    if a.shape[1] != n:
        out = np.full((len(df), n), np.nan, dtype=float)
        m = min(n, a.shape[1])
        out[:, :m] = a[:, :m]
        return out
    return a

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

def compute_point_metrics(labels, scores, higher_is_better=True):
    """
    Compute various classification and ranking metrics for virtual screening evaluation.
    
    Metrics computed:
    - ROC-AUC: Area Under the Receiver Operating Characteristic curve (standard AUC)
      Measures the ability to rank actives higher than inactives across all thresholds
    - PR-AUC: Area Under the Precision-Recall curve
      More informative for imbalanced datasets (typical in virtual screening)
    - BEDROC: Boltzmann-Enhanced Discrimination of ROC
      Emphasizes early recognition of actives
    - EF (Enrichment Factor): Ratio of actives found vs random selection at top X%
    - NEF (Normalized EF): EF normalized by maximum possible enrichment
    
    Args:
        labels: Binary labels (1=active, 0=inactive)
        scores: Predicted scores (interpretation depends on higher_is_better)
        higher_is_better: If True, higher scores indicate better compounds
    
    Returns:
        Dictionary of metric values
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    
    valid_mask = ~np.isnan(scores)
    if np.sum(valid_mask) < 2:
        return {"ROC-AUC": np.nan, "PR-AUC": np.nan, "BEDROC(20)": np.nan,
                "NEF1%": np.nan, "NEF5%": np.nan, "NEF10%": np.nan, 
                "EF1%": np.nan, "EF5%": np.nan, "EF10%": np.nan}
    labels = labels[valid_mask]
    scores = scores[valid_mask]

    if np.unique(labels).size < 2:
        return {"ROC-AUC": np.nan, "PR-AUC": np.nan, "BEDROC(20)": np.nan,
                "NEF1%": np.nan, "NEF5%": np.nan, "NEF10%": np.nan,
                "EF1%": np.nan, "EF5%": np.nan, "EF10%": np.nan}
    
    # ROC-AUC: Standard AUC metric for classification
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
    except Exception:
        roc_auc = np.nan
    
    # PR-AUC: Precision-Recall AUC (better for imbalanced data)
    # Note: sklearn's precision_recall_curve returns (precision, recall, thresholds)
    # where precision[i] is the precision at threshold[i] and recall[i] is the recall
    try:
        prec, rec, _ = precision_recall_curve(labels, scores)
        # Compute AUC of the precision-recall curve
        # This is correct: we integrate precision over recall
        pr_auc = auc(rec, prec)
    except Exception:
        pr_auc = np.nan
    
    bed, nef1, nef5, nef10 = np.nan, np.nan, np.nan, np.nan
    ef1, ef5, ef10 = np.nan, np.nan, np.nan
    
    if RDScoring is not None:
        if not higher_is_better:
            scores = -scores
        arr = [[float(s), bool(y)] for s,y in zip(scores, labels)]
        try:
            bed = float(RDScoring.CalcBEDROC(arr, 1, 20.0))
        except Exception:
            pass
        try:
            efs = RDScoring.CalcEnrichment(arr, 1, [0.01, 0.05, 0.10])
            ef1, ef5, ef10 = float(efs[0]), float(efs[1]), float(efs[2])
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
            pass

    return {"ROC-AUC": float(roc_auc), "PR-AUC": float(pr_auc), "BEDROC(20)": float(bed),
            "NEF1%": float(nef1), "NEF5%": float(nef5), "NEF10%": float(nef10),
            "EF1%": float(ef1), "EF5%": float(ef5), "EF10%": float(ef10)}

def euclidean_dist_matrix(vec_ref, mat):
    if vec_ref is None or np.isnan(vec_ref).any():
        return np.full((mat.shape[0],), np.nan, dtype=float)
    return np.linalg.norm(mat - vec_ref, axis=1)

def bootstrap_ci(values, n_bootstrap=1000, ci=95, statistic=np.mean):
    """Calculate bootstrap confidence interval using percentile method."""
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return np.nan, np.nan
    
    np.random.seed(42)
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_stats.append(statistic(sample))
    
    lower = np.percentile(bootstrap_stats, (100 - ci) / 2)
    upper = np.percentile(bootstrap_stats, 100 - (100 - ci) / 2)
    return lower, upper

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
        print(f"Cache not found. Building from {master_csv}...")
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
            chunk_iter = pd.read_csv(master_csv, dtype=str, keep_default_na=False, chunksize=chunksize)

        n_workers = cpu_count()
        worker_func = partial(process_chunk, usr_cols=USR_cols, 
                              usrcat_cols=USRCAT_cols, es_cols=ES_cols)
        results = process_map(worker_func, chunk_iter, 
                              max_workers=n_workers, chunksize=1, desc="Processing CSV")
        
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
        except Exception as e:
            print(f"Warning: Failed for target {t}: {e}", file=sys.stderr)
            ref_desc[t] = (None, None, None)

    print("Building scores per compound...")
    rows = []
    for i in tqdm(range(len(ids)), desc="Building scores"):
        uref, ucr, esr = ref_desc.get(targets[i], (None,None,None))
        rows.append({
            "id": ids[i],
            "smiles": smiles[i],
            "is_active": int(labels[i]),
            "Protein_ID": targets[i],
            "USR_score": float(euclidean_dist_matrix(uref, usr[i:i+1])[0]),
            "USRCAT_score": float(euclidean_dist_matrix(ucr, usrcat[i:i+1])[0]),
            "Electroshape_score": float(euclidean_dist_matrix(esr, es[i:i+1])[0]),
        })
    scores_df = pd.DataFrame(rows)
    scores_csv = os.path.join(outdir, "scores.csv")
    scores_df.to_csv(scores_csv, index=False)
    print("Wrote", scores_csv)

    print("Calculating per-target metrics...")
    methods = {
        "USR": ("USR_score", False),
        "USRCAT": ("USRCAT_score", False),
        "Electroshape": ("Electroshape_score", False)
    }
    per_rows = []
    for t in tqdm(unique_targets, desc="Per-target metrics"):
        msk = (scores_df["Protein_ID"] == t)
        sub = scores_df.loc[msk]
        lab = sub["is_active"].astype(int).values 
        row = {"Protein_ID": t, "N_Actives": int(lab.sum()), "N_Compounds": int(msk.sum())}
        for mname, (col, hib) in methods.items():
            vals = sub[col].astype(float).values
            metrics = compute_point_metrics(lab, vals, higher_is_better=hib)
            for k,v in metrics.items():
                row[f"{mname}_{k}"] = v
        per_rows.append(row)
    per_df = pd.DataFrame(per_rows)
    per_df.to_csv(os.path.join(outdir, "per_target_metrics.csv"), index=False)
    print("Wrote per_target_metrics.csv")

    print("Calculating global metrics with bootstrap CIs and pairwise p-values...")
    metric_names = ["ROC-AUC", "PR-AUC", "BEDROC(20)", "NEF1%", "NEF5%", "NEF10%", 
                    "EF1%", "EF5%", "EF10%"]
    
    # Collect data for each method
    method_data = {}
    for mname in methods.keys():
        method_data[mname] = {}
        for metric in metric_names:
            val_col = f"{mname}_{metric}"
            if val_col in per_df.columns:
                vals = per_df[val_col].dropna().values
                method_data[mname][metric] = vals
    
    # Build enhanced table with methods as columns and pairwise comparisons
    method_names = list(methods.keys())
    table_rows = []
    
    for metric in metric_names:
        row = {"Metric": metric}
        
        # Add median and CI for each method
        for mname in methods.keys():
            if metric in method_data[mname]:
                vals = method_data[mname][metric]
                if len(vals) > 0:
                    median_val = np.median(vals)
                    ci_low, ci_high = bootstrap_ci(vals, n_bootstrap=1000, ci=95, statistic=np.median)
                    row[mname] = f"{median_val:.3f} [{ci_low:.3f}-{ci_high:.3f}]"
                else:
                    row[mname] = "N/A"
            else:
                row[mname] = "N/A"
        
        # Add pairwise comparison columns (p-values and significance)
        for i, method1 in enumerate(method_names):
            for method2 in method_names[i+1:]:
                comparison_name = f"{method1}_vs_{method2}"
                if metric in method_data[method1] and metric in method_data[method2]:
                    vals1 = method_data[method1][metric]
                    vals2 = method_data[method2][metric]
                    
                    if len(vals1) > 0 and len(vals2) > 0:
                        try:
                            stat, pval = mannwhitneyu(vals1, vals2, alternative='two-sided')
                            row[f"{comparison_name}_pvalue"] = f"{pval:.4f}"
                            row[f"{comparison_name}_Significant"] = "Yes" if pval < 0.05 else "No"
                        except Exception as e:
                            row[f"{comparison_name}_pvalue"] = "N/A"
                            row[f"{comparison_name}_Significant"] = "N/A"
                    else:
                        row[f"{comparison_name}_pvalue"] = "N/A"
                        row[f"{comparison_name}_Significant"] = "N/A"
                else:
                    row[f"{comparison_name}_pvalue"] = "N/A"
                    row[f"{comparison_name}_Significant"] = "N/A"
        
        table_rows.append(row)
    
    global_df = pd.DataFrame(table_rows)
    global_df.to_csv(os.path.join(outdir, "global_metrics.csv"), index=False)
    print("Wrote global_metrics.csv with pairwise p-values and significance tests")
    
    # Also save the detailed Mann-Whitney tests in a separate file
    print("Saving detailed Mann-Whitney U tests...")
    method_names = list(methods.keys())
    mw_results = []
    
    for metric in metric_names:
        for i, method1 in enumerate(method_names):
            for method2 in method_names[i+1:]:
                if metric in method_data[method1] and metric in method_data[method2]:
                    vals1 = method_data[method1][metric]
                    vals2 = method_data[method2][metric]
                    
                    if len(vals1) > 0 and len(vals2) > 0:
                        try:
                            stat, pval = mannwhitneyu(vals1, vals2, alternative='two-sided')
                            mw_results.append({
                                "Metric": metric,
                                "Method_1": method1,
                                "Method_2": method2,
                                "Median_1": np.median(vals1),
                                "Median_2": np.median(vals2),
                                "U_statistic": stat,
                                "p_value": pval,
                                "Significant": "Yes" if pval < 0.05 else "No"
                            })
                        except:
                            pass
    
    if mw_results:
        mw_df = pd.DataFrame(mw_results)
        mw_df.to_csv(os.path.join(outdir, "mann_whitney_tests.csv"), index=False)
        print("Wrote mann_whitney_tests.csv")

    print("Running plots.py...")
    try:
        subprocess.run(["python3", os.path.join(base, "plots.py"), base], check=True)
    except subprocess.CalledProcessError as e:
        print("plots.py failed:", e, file=sys.stderr)
    except FileNotFoundError:
        print("plots.py not found. Skipping plotting.", file=sys.stderr)

if __name__ == "__main__":
    main()