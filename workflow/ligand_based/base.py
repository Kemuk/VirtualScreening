from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import polars as pl


class LigandBasedMethod(ABC):
    """Two-stage ligand-based screening method.

    Stage 1: generate_features_batch() — per-ligand feature vectors (called by ligand_features worker)
    Stage 2: compute_similarity()      — row-wise similarity (called by ligand_score worker)

    Contract for generate_features_batch():
    - Returns a pl.Series of fixed-size Array dtype:
        pl.Array(pl.UInt8, n_bytes)  for packed-bit 2D fingerprints
        pl.Array(pl.Float32, dim)    for 3D shape descriptors
    - Null entries represent failures (molecule could not be processed)
    - Use map_elements / group_by().map_groups() — do NOT call .to_list()
    - Because pl.Array is fixed-size, the scorer can call col.to_numpy()
      to get a 2D numpy matrix directly, with no custom helper.
    """

    requires_conformers: bool = False
    embedding_type: str = "2D"        # "2D" | "3D"
    default_metric: str = "tanimoto"  # "tanimoto" | "manhattan"

    @abstractmethod
    def generate_features_batch(
        self,
        df: pl.DataFrame,
        work_dir: Path,
        cfg: dict,
    ) -> pl.Series:
        """Generate feature vectors for a slice DataFrame.

        Args:
            df:       Slice DataFrame containing at least 'smiles_canonical'
                      and (for 3D methods) 'protein_id' columns.
            work_dir: Root directory for per-target intermediates.
            cfg:      ligand_based section from config.yaml.

        Returns:
            pl.Series of pl.Array dtype, same length as df, nulls on failure.
        """

    @abstractmethod
    def compute_similarity(
        self,
        db_mat: np.ndarray,
        template_mat: np.ndarray,
        cfg: dict,
    ) -> np.ndarray:
        """Row-wise similarity: db_mat[i] vs template_mat[i].

        Args:
            db_mat:      (N, D) numpy array of ligand feature vectors.
            template_mat:(N, D) numpy array of template feature vectors.
            cfg:         ligand_based config dict.

        Returns:
            1-D numpy array of shape (N,) with similarity scores in [0, 1].
        """
