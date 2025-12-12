#!/usr/bin/env python3
"""
worker_stream.py

Compute per-molecule descriptors (USR, USRCAT, Electroshape) averaged over conformers.
Gets USRCAT via rdMolDescriptors.GetUSRCAT(m) and handles 1D or 2D replies.

Usage:
  python worker_stream.py --input-fifo /path/to/slice.csv --out-csv part.csv --n-jobs 4 --n-confs 3
"""
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

# electroshape import left to fail if missing
from oddt import toolkit
from oddt.shape import electroshape

USR_N = 12
USRCAT_N = 60
ES_N = 15
INT32_MAX = 2_147_483_647

def safe_seed_from_smiles(smiles: str) -> int:
    return int(abs(hash(smiles)) % INT32_MAX)

def embed_conformers(mol, n_confs: int, seed: int):
    """Embed and optimize conformers; return molecule with Hs and list of conf ids."""
    m = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed) % INT32_MAX
    try:
        cids = list(AllChem.EmbedMultipleConfs(m, numConfs=n_confs, params=params))
    except Exception:
        cid = AllChem.EmbedMolecule(m, params)
        cids = [cid]
    # try to optimize; allow exceptions to propagate
    good = []
    for c in cids:
        AllChem.MMFFOptimizeMolecule(m, confId=int(c))
        good.append(int(c))
    return m, good

def usr_for_conf(m, conf_id):
    return np.array(rdMolDescriptors.GetUSR(m, int(conf_id)), dtype=float)

def usrcat_for_conf(m, conf_id):
    """
    Use rdMolDescriptors.GetUSRCAT(m) (no conf-id argument).
    The function may return:
      - a 1D sequence (length USRCAT_N) -> use it
      - a 2D-like sequence (n_conf x USRCAT_N) -> pick the row corresponding to conf_id
    If selection fails it raises an informative exception.
    """
    val = rdMolDescriptors.GetUSRCAT(m)  # call without conf id
    # convert to numpy
    arr = np.array(val, dtype=float)
    if arr.ndim == 1:
        if arr.size != USRCAT_N:
            raise ValueError(f"GetUSRCAT returned length {arr.size}, expected {USRCAT_N}")
        return arr
    elif arr.ndim == 2:
        # arr shape (n_conf, USRCAT_N). Need to map conf_id to index.
        # RDKit's ordering normally matches the molecule conformer ordering (0..n-1).
        # Find index of conf_id among m.GetConformers()
        conf_ids = [int(c.GetId()) for c in m.GetConformers()]
        try:
            idx = conf_ids.index(int(conf_id))
        except ValueError:
            # fallback: if conf_id > len(arr) use nearest index
            if 0 <= int(conf_id) < arr.shape[0]:
                idx = int(conf_id)
            else:
                raise ValueError(f"conf_id {conf_id} not found among molecule conformers {conf_ids}")
        row = arr[idx]
        if row.size != USRCAT_N:
            raise ValueError(f"GetUSRCAT returned per-conf length {row.size}, expected {USRCAT_N}")
        return row
    else:
        raise ValueError(f"GetUSRCAT returned array with ndim={arr.ndim}; expected 1 or 2")

def electroshape_for_conf(m, conf_id):
    mb = Chem.MolToMolBlock(m, confId=int(conf_id))
    oddt_m = toolkit.readstring('sdf', mb)
    desc = electroshape(oddt_m)
    return np.array(desc, dtype=float)

def avg_or_nan(list_of_arrays, length):
    if not list_of_arrays:
        return np.full(length, np.nan, dtype=float)
    stacked = np.vstack(list_of_arrays)
    return np.nanmean(stacked, axis=0)

def process_record(rec: dict, n_confs: int):
    # let errors propagate so you see them
    smiles = (rec.get('smiles') or rec.get('SMILES') or '').strip()
    if not smiles:
        return None
    ident = (rec.get('id') or rec.get('name') or '').strip()
    protein = (rec.get('Protein_ID') or rec.get('ProteinId') or rec.get('target') or '').strip()
    ref = (rec.get('ref_ligand_path') or '').strip()
    label = int(rec.get('label', 0))

    m0 = Chem.MolFromSmiles(smiles)
    if m0 is None:
        raise ValueError(f"Invalid SMILES for id={ident}: {smiles}")

    seed = safe_seed_from_smiles(smiles)
    m3d, conf_ids = embed_conformers(m0, n_confs, seed)

    usr_list = []
    usrcat_list = []
    es_list = []
    for cid in conf_ids:
        usr_list.append(usr_for_conf(m3d, cid))
        usrcat_list.append(usrcat_for_conf(m3d, cid))
        es_list.append(electroshape_for_conf(m3d, cid))

    usr_avg = avg_or_nan(usr_list, USR_N)
    usrcat_avg = avg_or_nan(usrcat_list, USRCAT_N)
    es_avg = avg_or_nan(es_list, ES_N)

    out = [ident, smiles, label, protein, ref]
    out.extend([float(x) for x in usr_avg])
    out.extend([float(x) for x in usrcat_avg])
    out.extend([float(x) for x in es_avg])
    return out

def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

def main():
    p = argparse.ArgumentParser(description="Worker: compute descriptors. Uses GetUSRCAT(m).")
    p.add_argument('--input-fifo', required=True)
    p.add_argument('--out-csv', required=True)
    p.add_argument('--n-jobs', type=int, default=1)
    p.add_argument('--n-confs', type=int, default=3)
    args = p.parse_args()

    df = pd.read_csv(args.input_fifo, dtype=str, keep_default_na=False)
    if df.shape[0] == 0:
        open(args.out_csv, 'w').close()
        return

    records = df.to_dict(orient='records')
    results = []

    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        futures = {ex.submit(process_record, rec, args.n_confs): i for i, rec in enumerate(records)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="processing", unit="mol"):
            row = fut.result()    # let exceptions propagate
            if row is not None:
                results.append(row)

    if not results:
        open(args.out_csv, 'w').close()
        return

    header = ['id', 'smiles', 'label', 'Protein_ID', 'ref_ligand_path']
    header += [f'USR_{i}' for i in range(USR_N)]
    header += [f'USRCAT_{i}' for i in range(USRCAT_N)]
    header += [f'ES_{i}' for i in range(ES_N)]

    write_csv(args.out_csv, header, results)

if __name__ == '__main__':
    main()
