#!/usr/bin/env python3
"""
make_manifest_from_vina.py

Write ligand_based/manifest.csv with columns:
  id,smiles,label,Protein_ID,ref_ligand_path

Uses only actives_smile and inactives_smile from ../LIT_PCBA/vina_boxes.csv.
"""
import argparse, os, sys, csv
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def read_simple_smi(path, target, prefix, label, ref_ligand):
    rows = []
    with open(path, 'r') as fh:
        for i, line in enumerate(fh):
            s = line.strip()
            if not s:
                continue
            parts = s.split(None, 1)
            smiles = parts[0]
            ident = parts[1].strip() if len(parts) > 1 else f"{target}_{prefix}_{i:06d}"
            rows.append((ident, smiles, label, target, ref_ligand))
    return rows

def process_row_simple(row):
    tgt = row['target']
    # derive ref ligand path from receptor_pdbqt
    rp = row.get('receptor_pdbqt','').strip()
    if rp:
        ref = rp.replace('_protein.pdbqt', '_ligand.mol2')
    else:
        ref = ''
    out = []
    a = row.get('actives_smile', '').strip()
    i = row.get('inactives_smile', '').strip()
    if a and os.path.exists(a):
        out.extend(read_simple_smi(a, tgt, 'A', 1, ref))
    if i and os.path.exists(i):
        out.extend(read_simple_smi(i, tgt, 'I', 0, ref))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vina-csv', default=os.path.join('..','LIT_PCBA','vina_boxes.csv'))
    p.add_argument('--out', default='manifest.csv')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    if os.path.exists(args.out) and not args.force:
        sys.exit(f"{args.out} exists. Use --force to overwrite.")
    if not os.path.exists(args.vina_csv):
        sys.exit(f"vina_boxes.csv not found: {args.vina_csv}")

    df = pd.read_csv(args.vina_csv, dtype=str, keep_default_na=False)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_row_simple, df.iloc[r].to_dict()): r for r in range(len(df))}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                rows.extend(res)

    if not rows:
        sys.exit("no SMILES collected (check actives/inactives paths in vina_boxes.csv)")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['id','smiles','label','Protein_ID','ref_ligand_path'])
        for ident, smiles, label, protein, ref in rows:
            w.writerow([ident, smiles, label, protein, ref])

    print(f"Wrote {len(rows)} rows -> {args.out}")

if __name__ == '__main__':
    main()
