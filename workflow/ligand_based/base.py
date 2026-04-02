from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class LigandBasedMethod(ABC):
    """Score a list of SMILES against a template molecule for one target."""

    @abstractmethod
    def run_target(
        self,
        smiles: list,
        template_smiles: str,
        work_dir: Path,
        cfg: dict,
    ) -> pd.Series:
        """
        Args:
            smiles: Canonical SMILES for all ligands in this target.
            template_smiles: Query molecule (must appear in smiles list).
            work_dir: Per-target working directory for intermediates.
            cfg: ligand_based config section from config.yaml.

        Returns:
            pd.Series of float32 similarity scores, indexed 0..N-1,
            aligned to the input smiles list.
        """
