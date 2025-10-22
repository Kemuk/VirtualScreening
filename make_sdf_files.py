#!/usr/bin/env python3
"""
Convert docked Vina PDBQTs → SDFs using OpenBabel and RDKit.
Reads docked_vina and docked_sdf locations from vina_boxes.csv.
Expands brace patterns like {actives,inactives}.
Keeps only the pose with the lowest binding score (most negative kcal/mol).
VALIDATES and CORRECTS nitrogen valence issues using RDKit.
Parallelized with ProcessPoolExecutor.
"""

import sys, re, argparse, subprocess, tempfile, os
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm

# <-- NEW: Add RDKit import. You must have rdkit installed (`pip install rdkit`).
from rdkit import Chem

# ---------- utils ----------
def which(cmd: str) -> str:
    from shutil import which as _which
    return _which(cmd) or ""

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def split_outside_braces(s: str) -> List[str]:
    """Split by commas, ignoring commas inside braces { }."""
    out, cur, depth = [], [], 0
    for ch in s:
        if ch == ',' and depth == 0:
            out.append(''.join(cur)); cur = []
        else:
            if ch == '{': depth += 1
            elif ch == '}' and depth > 0: depth -= 1
            cur.append(ch)
    out.append(''.join(cur))
    return out

def brace_items(pattern: str) -> List[str]:
    """Return item list inside first {...}. If none, return [] (meaning no expansion)."""
    m = re.search(r"\{([^}]*)\}", pattern)
    if not m:
        return []
    return [s.strip() for s in m.group(1).replace("|", ",").split(",") if s.strip()]

def brace_replace(pattern: str, item: str) -> str:
    """Replace first {...} in pattern with a single item; if none, return pattern."""
    return re.sub(r"\{[^}]*\}", item, pattern, count=1)

def load_boxes(csv_name: str) -> List[Dict[str, str]]:
    """Load vina_boxes.csv in CWD (brace-safe)."""
    p = Path(csv_name)
    if not p.exists():
        sys.exit(f"ERROR: Could not find {csv_name}")
    lines = p.read_text(encoding="utf-8").splitlines()
    if not lines:
        sys.exit(f"ERROR: CSV {p} is empty.")
    header = split_outside_braces(lines[0])
    req = {"target", "docked_vina", "docked_sdf"}
    if not req.issubset(header):
        missing = req - set(header)
        sys.exit(f"ERROR: CSV {p} missing columns: {', '.join(sorted(missing))}")
    rows = []
    for line in lines[1:]:
        cols = split_outside_braces(line)
        if len(cols) != len(header):
            if len(cols) < len(header):
                cols += [""] * (len(header) - len(cols))
            else:
                cols = cols[:len(header)]
        rows.append(dict(zip(header, cols)))
    return rows

# <-- NEW: Our RDKit chemical fixer function from the previous discussion.
def fix_chemical_structure(mol):
    """
    Fixes common chemical representation errors in RDKit molecule objects,
    specifically focusing on mis-defined nitrogen groups.
    """
    try:
        rw_mol = Chem.RWMol(mol)
        nitro_pattern = Chem.MolFromSmarts('[N;v5](=O)=O')
        matches = rw_mol.GetSubstructMatches(nitro_pattern)

        if matches:
            for match in matches:
                nitro_n = rw_mol.GetAtomWithIdx(match[0])
                for neighbor in nitro_n.GetNeighbors():
                    if neighbor.GetSymbol() == 'O':
                        bond = rw_mol.GetBondBetweenAtoms(nitro_n.GetIdx(), neighbor.GetIdx())
                        if bond.GetBondType() == Chem.BondType.DOUBLE:
                            bond.SetBondType(Chem.BondType.SINGLE)
                            nitro_n.SetFormalCharge(1)
                            neighbor.SetFormalCharge(-1)
                            break
        
        for atom in rw_mol.GetAtoms():
            if atom.GetSymbol() == 'N' and atom.GetExplicitValence() == 4 and atom.GetFormalCharge() == 0:
                atom.SetFormalCharge(1)

        final_mol = rw_mol.GetMol()
        Chem.SanitizeMol(final_mol)
        return final_mol
    except Exception:
        return None

# <-- MODIFIED: This entire function is replaced with a new one that uses RDKit.
def process_pdbqt_to_sdf(pdbqt_in: Path, sdf_out: Path, obabel_bin: str = "obabel") -> bool:
    """
    Convert PDBQT → SDF using OpenBabel, select the best pose,
    then use RDKit to validate, fix, and write the final SDF.
    """
    if not which(obabel_bin):
        sys.exit(f"ERROR: {obabel_bin} not found in PATH")

    sdf_tmp = Path(tempfile.mktemp(prefix="obabel_", suffix=".sdf"))
    try:
        # Step 1: Use OpenBabel to convert all poses from PDBQT to a temporary SDF.
        r = subprocess.run(
            [obabel_bin, "-ipdbqt", str(pdbqt_in), "-osdf", "-O", str(sdf_tmp)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if r.returncode != 0 or not sdf_tmp.exists():
            return False

        # Step 2: Parse the temporary SDF to find the pose with the best Vina score.
        with open(sdf_tmp, "r") as f:
            blocks = f.read().split("$$$$\n")

        best_block, best_score = None, float("inf")
        for block in blocks:
            if not block.strip():
                continue
            m = re.search(r"VINA RESULT:\s*([-\d.]+)", block)
            if m:
                score = float(m.group(1))
                if score < best_score:
                    best_score = score
                    best_block = block

        # Step 3: If a best pose was found, process it with RDKit.
        if best_block:
            # RDKit can read a molecule from a text block. We load it without sanitizing first.
            suppl = Chem.SDMolSupplier()
            suppl.SetData(best_block.strip() + "\n$$$$\n", sanitize=False)
            mol = next(suppl)

            if not mol:
                return False # RDKit couldn't parse the block from OpenBabel.

            # Apply our chemical fixer function.
            fixed_mol = fix_chemical_structure(mol)

            if fixed_mol:
                ensure_dir(sdf_out.parent)
                # Write the final, corrected molecule to the destination SDF file.
                with Chem.SDWriter(str(sdf_out)) as writer:
                    # RDKit will preserve properties like the Vina score.
                    writer.write(fixed_mol)
                return True
        
        return False
    finally:
        try:
            sdf_tmp.unlink(missing_ok=True)
        except Exception:
            pass

# ---------- worker ----------
def worker(task: Tuple[str, Path, Path, str]) -> Tuple[str, Path]:
    target, pdbqt, sdf_out, obabel_bin = task
    # <-- MODIFIED: Call the new function.
    ok = process_pdbqt_to_sdf(pdbqt, sdf_out, obabel_bin)
    return ("ok" if ok else "fail", pdbqt)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Convert docked PDBQTs → best-pose SDFs using vina_boxes.csv, OpenBabel, and RDKit.")
    ap.add_argument("--csv", default="vina_boxes.csv", help="CSV file (default: vina_boxes.csv)")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers (default: all cores)")
    ap.add_argument("--obabel", default="obabel", help="Path to obabel binary")
    ap.add_argument("--only-target", action="append", default=[], help="Restrict to these targets (repeatable)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing SDF files.")
    args = ap.parse_args()

    rows = load_boxes(args.csv)
    if args.only_target:
        keep = set(args.only_target)
        rows = [r for r in rows if r.get("target", "").strip() in keep]

    tasks, skipped_count = [], 0
    for row in rows:
        target = row["target"].strip()
        dv_pat = row["docked_vina"].strip()
        ds_pat = row["docked_sdf"].strip()
        items = brace_items(dv_pat) or [None]

        for item in items:
            dv_dir = Path(brace_replace(dv_pat, item) if item else dv_pat)
            ds_dir = Path(brace_replace(ds_pat, item) if item else ds_pat)
            if not dv_dir.exists():
                print(f"WARNING: Missing docked_vina dir {dv_dir}", file=sys.stderr)
                continue
            for pdbqt in dv_dir.glob("*.pdbqt"):
                lig_id = pdbqt.stem
                out_sdf = ds_dir / f"{lig_id}.sdf"
                
                if not args.overwrite and out_sdf.exists():
                    skipped_count += 1
                    continue

                tasks.append((target, pdbqt, out_sdf, args.obabel))

    if skipped_count > 0:
        print(f"Skipped {skipped_count} conversions for existing SDF files. Use --overwrite to re-process.")

    if not tasks:
        sys.exit("No new files to convert. Check vina_boxes.csv paths or use --overwrite.")

    print(f"Planned conversions: {len(tasks)} (targets: {len(set(t[0] for t in tasks))})")

    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="SDF conversion", unit="lig"):
            status, pdbqt = fut.result()
            if status == "ok":
                ok += 1
            else:
                fail += 1
                # Optional: Add a print statement to identify which files fail
                # print(f"Failed to process: {pdbqt}", file=sys.stderr)

    print(f"\nDone. OK: {ok}, Failed: {fail}")


if __name__ == "__main__":
    main()