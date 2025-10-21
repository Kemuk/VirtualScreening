from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from pathlib import Path
import pandas as pd
import csv

def canonicalize_smiles(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return Chem.MolToSmiles(m, isomericSmiles=True)

def load_smi_file(path):
    path = Path(path)
    if not path.exists():
        return []
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    # Allow files with columns: SMILES[,id]
    out = []
    for L in lines:
        parts = L.split()
        out.append(parts[0])
    return out

def preprocess_target(data_root, processed_root, target, cfg):
    data_root = Path(data_root)
    processed_root = Path(processed_root) / target
    processed_root.mkdir(parents=True, exist_ok=True)

    target_dir = data_root / target
    actives = load_smi_file(target_dir / "actives.smi")
    inactives = load_smi_file(target_dir / "inactives.smi")
    records = []
    seen = set()
    for label, smiles_list in (("active", actives), ("inactive", inactives)):
        for s in smiles_list:
            cs = canonicalize_smiles(s)
            if cs is None:
                continue
            if cs in seen:
                continue
            seen.add(cs)
            records.append({"smiles": cs, "label": label})
    df = pd.DataFrame(records)
    df.to_csv(processed_root / "cleaned_smiles.csv", index=False)
    # simple labels.csv (smiles,label)
    df[["smiles", "label"]].to_csv(processed_root / "labels.csv", index=False)
    return processed_root / "cleaned_smiles.csv"
