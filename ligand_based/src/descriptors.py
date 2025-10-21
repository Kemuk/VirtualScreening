from pathlib import Path
import numpy as np
import pandas as pd
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os

def _load_sdf_first_mol(sdf_path):
    from rdkit import Chem
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for m in suppl:
        if m is not None:
            return m
    return None

def _worker_compute(sdf_path):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    from rdkit.Chem import rdMolDescriptors
    m = _load_sdf_first_mol(sdf_path)
    if m is None:
        return {"file": str(sdf_path), "smiles": None, "usr": None, "usrcat": None, "error": "no_valid_mol"}
    try:
        orig = m.GetProp("_orig_smiles")
    except:
        orig = None
    usr_v = list(rdMolDescriptors.GetUSR(m))
    try:
        usrcat_v = list(rdMolDescriptors.GetUSRCAT(m))
    except Exception:
        usrcat_v = None
    return {"file": str(sdf_path), "smiles": orig, "usr": usr_v, "usrcat": usrcat_v, "error": None}

def compute_descr_for_target(processed_root, target, cfg):
    processed_root = Path(processed_root) / target
    conf_dir = processed_root / "conformers"
    if not conf_dir.exists():
        raise SystemExit("Conformers directory not found")

    sdf_files = sorted(conf_dir.glob("*.sdf"))
    if not sdf_files:
        raise SystemExit("No per-molecule SDF files found")

    max_workers = int(cfg.get("max_workers", 1))
    tqdm_disable = os.getenv("TQDM_DISABLE", "0") == "1"
    tqdm_pos = int(cfg.get("tqdm_position", 0))

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_worker_compute, str(p)): str(p) for p in sdf_files}
        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc=f"descriptors:{target}",
                        position=tqdm_pos,
                        leave=True,
                        disable=tqdm_disable):
            res = fut.result()
            results.append(res)

    usr_map = {}
    usrcat_map = {}
    for r in results:
        fname = Path(r["file"]).stem
        if r.get("usr"):
            usr_map[fname] = np.asarray(r["usr"], dtype=cfg.get("descriptor_dtype", "float32"))
        if r.get("usrcat"):
            usrcat_map[fname] = np.asarray(r["usrcat"], dtype=cfg.get("descriptor_dtype", "float32"))

    if usr_map:
        np.savez_compressed(processed_root / "usr_vectors.npz", **usr_map)
    if usrcat_map:
        np.savez_compressed(processed_root / "usrcat_vectors.npz", **usrcat_map)
    (processed_root / "descriptors_meta.json").write_text(json.dumps(results, indent=2))
    return str(processed_root / "usr_vectors.npz")
