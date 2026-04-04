import numpy as np
import polars as pl
from pathlib import Path

from ligand_based.src.conformers import generate_conformers_for_target, _safe_name
from ligand_based.src.descriptors import compute_descr_for_target
from workflow.ligand_based.base import LigandBasedMethod


class _USR3DBase(LigandBasedMethod):
    """Base for USR / USRCAT 3D shape-similarity methods.

    Subclasses set:
        _npz_name  — filename inside the target directory ("usr_vectors.npz" or "usrcat_vectors.npz")
        _dim       — feature vector length (12 for USR, 60 for USRCAT)
    """

    requires_conformers = True
    embedding_type = "3D"
    default_metric = "manhattan"

    _npz_name: str = None
    _dim: int = None

    def generate_features_batch(self, df: pl.DataFrame, work_dir: Path, cfg: dict) -> pl.Series:
        npz_name = self._npz_name
        dim = self._dim

        def _process_target(group_df: pl.DataFrame) -> pl.DataFrame:
            pid = group_df["protein_id"][0]
            target_dir = work_dir / pid
            target_dir.mkdir(parents=True, exist_ok=True)

            # Write cleaned_smiles.csv in the format expected by src helpers
            (group_df.select("smiles_canonical")
             .rename({"smiles_canonical": "smiles"})
             .write_csv(target_dir / "cleaned_smiles.csv"))

            generate_conformers_for_target(work_dir, pid, cfg)
            compute_descr_for_target(work_dir, pid, cfg)

            npz_dict = dict(np.load(target_dir / npz_name))

            # Vectorised per-row lookup via map_elements
            return group_df.with_columns(
                group_df["smiles_canonical"].map_elements(
                    lambda s, d=npz_dict: d[_safe_name(s)].tolist() if _safe_name(s) in d else None,
                    return_dtype=pl.Array(pl.Float32, dim),
                ).alias("feature_vec")
            )

        # group_by().map_groups() replaces the explicit per-target for-loop
        return (
            df.with_row_index("__idx")
            .group_by("protein_id")
            .map_groups(_process_target)
            .sort("__idx")
            .get_column("feature_vec")
        )

    def compute_similarity(self, db_mat: np.ndarray, template_mat: np.ndarray, cfg: dict) -> np.ndarray:
        dists = np.abs(db_mat - template_mat).sum(axis=1)
        return 1.0 / (1.0 + dists)


class USRMethod(_USR3DBase):
    """USR (Ultrafast Shape Recognition) — 12-dim, Manhattan distance."""

    _npz_name = "usr_vectors.npz"
    _dim = 12


class USRCATMethod(_USR3DBase):
    """USRCAT (USR with CREDO Atom Types) — 60-dim, Manhattan distance."""

    _npz_name = "usrcat_vectors.npz"
    _dim = 60
