import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import pairwise_distances

from ligand_based.src.conformers import generate_conformers_for_target, _safe_name
from ligand_based.src.descriptors import compute_descr_for_target
from ligand_based.src.index_search import _sim_from_dist
from workflow.ligand_based.base import LigandBasedMethod


class USRCATMethod(LigandBasedMethod):
    """
    Shape-based ligand similarity using USR/USRCAT descriptors.

    Wraps the existing ligand_based/src pipeline:
      conformers → USR/USRCAT descriptors → Manhattan distance → similarity score
    """

    def run_target(self, smiles, template_smiles, work_dir, cfg):
        target = work_dir.name
        work_dir.mkdir(parents=True, exist_ok=True)

        # Write cleaned_smiles.csv in the format expected by the existing src functions
        pd.DataFrame({"smiles": smiles}).to_csv(work_dir / "cleaned_smiles.csv", index=False)

        # Conformer generation and descriptor computation (both have tqdm progress internally)
        generate_conformers_for_target(work_dir.parent, target, cfg)
        compute_descr_for_target(work_dir.parent, target, cfg)

        # Load descriptor vectors; prefer USRCAT (60-dim) over USR (12-dim)
        which = "usrcat" if (work_dir / "usrcat_vectors.npz").exists() else "usr"
        data = np.load(work_dir / f"{which}_vectors.npz")

        # Build (N, D) embedding matrix preserving input order
        zero = np.zeros(data[list(data.files)[0]].shape, dtype="float32")
        keys = [_safe_name(s) for s in smiles]
        db_mat = np.vstack([data[k] if k in data else zero for k in keys])

        # Score all molecules against the template
        tmpl_idx = smiles.index(template_smiles) if template_smiles in smiles else 0
        query = db_mat[tmpl_idx : tmpl_idx + 1]
        dists = pairwise_distances(query, db_mat, metric="manhattan").reshape(-1)
        scores = _sim_from_dist(dists, cfg.get("similarity_conversion", "inv1_plus"))
        return pd.Series(scores, dtype="float32")
