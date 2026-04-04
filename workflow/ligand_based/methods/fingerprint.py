import numpy as np
import polars as pl
from pathlib import Path

from workflow.ligand_based.base import LigandBasedMethod


class ECFPMethod(LigandBasedMethod):
    """Morgan (ECFP4) circular fingerprint — 2D, Tanimoto similarity.

    Fingerprint bits are packed into uint8 bytes and stored as
    pl.Array(pl.UInt8, n_bytes) so the scorer can call .to_numpy()
    to get a 2D matrix and unpack with np.unpackbits.
    """

    requires_conformers = False
    embedding_type = "2D"
    default_metric = "tanimoto"

    def _fp(self, smiles: str, cfg: dict):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        n_bits = cfg.get("ecfp_nbits", 2048)
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=cfg.get("ecfp_radius", 2), nBits=n_bits
        )
        return np.packbits(np.array(fp)).tolist()

    def generate_features_batch(self, df: pl.DataFrame, work_dir: Path, cfg: dict) -> pl.Series:
        n_bytes = cfg.get("ecfp_nbits", 2048) // 8
        return df["smiles_canonical"].map_elements(
            lambda s: self._fp(s, cfg),
            return_dtype=pl.Array(pl.UInt8, n_bytes),
        )

    def compute_similarity(self, db_mat: np.ndarray, template_mat: np.ndarray, cfg: dict) -> np.ndarray:
        db   = np.unpackbits(db_mat,   axis=1).astype(np.float32)
        tmpl = np.unpackbits(template_mat, axis=1).astype(np.float32)
        intersection = (db * tmpl).sum(axis=1)
        union = (db + tmpl - db * tmpl).sum(axis=1)
        return np.where(union > 0, intersection / union, 0.0)


class MACCSMethod(LigandBasedMethod):
    """MACCS structural keys — 167-bit, 2D, Tanimoto similarity.

    Packed into 21 bytes (ceil(167/8)); the last byte has one padding bit.
    """

    requires_conformers = False
    embedding_type = "2D"
    default_metric = "tanimoto"

    _N_BYTES = 21  # ceil(167 / 8)

    def _fp(self, smiles: str, cfg: dict):
        from rdkit import Chem
        from rdkit.Chem import MACCSkeys

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.packbits(np.array(fp)).tolist()

    def generate_features_batch(self, df: pl.DataFrame, work_dir: Path, cfg: dict) -> pl.Series:
        return df["smiles_canonical"].map_elements(
            lambda s: self._fp(s, cfg),
            return_dtype=pl.Array(pl.UInt8, self._N_BYTES),
        )

    def compute_similarity(self, db_mat: np.ndarray, template_mat: np.ndarray, cfg: dict) -> np.ndarray:
        db   = np.unpackbits(db_mat,   axis=1).astype(np.float32)
        tmpl = np.unpackbits(template_mat, axis=1).astype(np.float32)
        intersection = (db * tmpl).sum(axis=1)
        union = (db + tmpl - db * tmpl).sum(axis=1)
        return np.where(union > 0, intersection / union, 0.0)
