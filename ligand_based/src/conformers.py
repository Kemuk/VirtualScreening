from pathlib import Path
import hashlib
import pandas as pd
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os

def _safe_name(smiles: str) -> str:
    h = hashlib.sha1(smiles.encode("utf8")).hexdigest()[:12]
    return f"{h}"

def _worker_generate(args):
    smiles, out_dir, n_conf, seed, idx = args
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        m0 = Chem.MolFromSmiles(smiles)
        if m0 is None:
            return {"smiles": smiles, "n_confs": 0, "file": None, "error": "bad_smiles"}
        m = Chem.AddHs(m0)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed + idx)
        params.pruneRmsThresh = 0.5
        cids = AllChem.EmbedMultipleConfs(m, numConfs=int(n_conf), params=params)
        if len(cids) == 0:
            return {"smiles": smiles, "n_confs": 0, "file": None, "error": "no_confs"}
        AllChem.MMFFOptimizeMoleculeConfs(m, numThreads=1)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / (f"{_safe_name(smiles)}.sdf")
        w = Chem.SDWriter(str(fname))
        for ci in cids:
            mol_copy = Chem.Mol(m)
            mol_copy.SetProp("_orig_smiles", smiles)
            mol_copy.SetProp("_Name", f"{_safe_name(smiles)}_conf{ci}")
            w.write(mol_copy, confId=int(ci))
        w.close()
        return {"smiles": smiles, "n_confs": len(cids), "file": str(fname), "error": None}
    except Exception as e:
        return {"smiles": smiles, "n_confs": 0, "file": None, "error": str(e)}

def generate_conformers_for_target(processed_root, target, cfg):
    processed_root = Path(processed_root) / target
    processed_root.mkdir(parents=True, exist_ok=True)
    conf_dir = processed_root / "conformers"
    conf_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_root / "cleaned_smiles.csv")
    smiles_list = df["smiles"].tolist()
    n_conf = int(cfg.get("conformers_per_mol", 3))
    seed = int(cfg.get("random_seed", 42))
    max_workers = int(cfg.get("max_workers", 1))
    tqdm_disable = os.getenv("TQDM_DISABLE", "0") == "1"
    tqdm_pos = int(cfg.get("tqdm_position", 0))

    tasks = [(smi, str(conf_dir), n_conf, seed, i) for i, smi in enumerate(smiles_list)]
    records = []
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_worker_generate, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc=f"conformers:{target}",
                        position=tqdm_pos,
                        leave=True,
                        disable=tqdm_disable):
            res = fut.result()
            records.append(res)

    meta_path = processed_root / "conformers_meta.json"
    meta_path.write_text(json.dumps(records, indent=2))
    pd.DataFrame(records).to_csv(processed_root / "conformer_index.csv", index=False)
    return str(processed_root / "conformer_index.csv")
