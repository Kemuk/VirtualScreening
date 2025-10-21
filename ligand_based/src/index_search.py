# src/index_search.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import pairwise_distances

def _load_vectors(processed_root, target, which="usrcat"):
    p = Path(processed_root) / target
    file = p / (f"{which}_vectors.npz")
    data = np.load(file)
    out = {k: data[k] for k in data.files}
    return out

def _sim_from_dist(d, mode="inv1_plus"):
    if mode == "inv1_plus":
        return 1.0 / (1.0 + d)
    else:
        dmax = d.max()
        return np.maximum(0, 1 - d / (dmax + 1e-12))

def rank_by_usrcat(processed_root, results_root, target, cfg, manifest):
    processed_root = Path(processed_root)
    results_root = Path(results_root) / target
    results_root.mkdir(parents=True, exist_ok=True)

    # load mapping of per-molecule file keys -> canonical SMILES
    conf_index = pd.read_csv(processed_root / target / "conformer_index.csv")
    # conformer_index.csv rows: smiles,file,n_confs,...
    # file expected like ".../conformers/<hash>.sdf" or "<hash>.sdf"
    conf_index["key"] = conf_index["file"].apply(lambda p: Path(p).stem if pd.notna(p) else None)
    key_to_smiles = dict(zip(conf_index["key"], conf_index["smiles"]))

    # load vectors
    which = "usrcat" if (processed_root / target / "usrcat_vectors.npz").exists() else "usr"
    vecs = _load_vectors(processed_root, target, which=which)
    keys = list(vecs.keys())
    db_keys = np.array(keys)
    db_mat = np.vstack([vecs[k] for k in db_keys])

    # select query vector (manifest template_ligand may be a SMILES or file key)
    template = manifest.get("template_ligand")
    # try match by canonical SMILES first
    query_vec = None
    if template:
        # if template equals a canonical SMILES in mapping, find corresponding key
        matches = [k for k, s in key_to_smiles.items() if s == template]
        if matches:
            qkey = matches[0]
            if qkey in vecs:
                query_vec = vecs[qkey]
    if query_vec is None:
        # fallback pick first vector
        query_vec = vecs[keys[0]]

    q = np.asarray(query_vec).reshape(1, -1)
    dists = pairwise_distances(q, db_mat, metric="manhattan").reshape(-1)
    sims = _sim_from_dist(dists, cfg.get("similarity_conversion", "inv1_plus"))
    ranked = pd.DataFrame({"key": db_keys, "score": sims})
    # map keys back to SMILES
    ranked["smiles"] = ranked["key"].map(key_to_smiles)
    # bring labels if present
    labels_path = processed_root / target / "labels.csv"
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        ranked = ranked.merge(labels, on="smiles", how="left")
    # order and output
    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
    out_csv = f"ranked_{which}_{target}.csv"
    ranked.to_csv(results_root / out_csv, index=False)
    return out_csv
