#!/usr/bin/env python3
"""
smi2pdbqt.py

Convert SMILES to PDBQT format for molecular docking.

Process:
  1. Read SMILES string
  2. Generate 3D coordinates using RDKit
  3. Add hydrogens
  4. Optimize geometry
  5. Convert to PDBQT via OpenBabel
"""

import argparse
import sys
import subprocess
import tempfile
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


def smiles_to_mol_with_3d(
    smiles: str,
    optimize: bool = True,
    max_iters: int = 200,
) -> Chem.Mol:
    """Convert SMILES to RDKit Mol with 3D coordinates."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"ERROR: Invalid SMILES: {smiles}", file=sys.stderr)
        return None

    mol = Chem.AddHs(mol)

    try:
        result = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=False)
        if result != 0:
            print(f"ERROR: 3D coordinate generation failed (code: {result})", file=sys.stderr)
            return None
    except Exception as e:
        print(f"ERROR: 3D embedding failed: {e}", file=sys.stderr)
        return None

    if optimize:
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
        except Exception as e:
            print(f"WARNING: MMFF optimization failed: {e}", file=sys.stderr)

    return mol


def sdf_to_pdbqt(
    sdf_path: Path,
    pdbqt_path: Path,
    ph: float = 7.4,
    partial_charge: str = "gasteiger",
) -> bool:
    """Convert SDF to PDBQT using OpenBabel."""
    pdbqt_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "obabel",
        str(sdf_path),
        "-opdbqt",
        "--gen3d",
        "-O", str(pdbqt_path),
        "-p", str(ph),
        "--partialcharge", partial_charge,
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: OpenBabel conversion failed", file=sys.stderr)
        print(f"Command: {' '.join(cmd)}", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("ERROR: obabel not found. Install OpenBabel.", file=sys.stderr)
        return False


def smiles_to_pdbqt(
    smiles: str,
    pdbqt_path: Path,
    ph: float = 7.4,
    partial_charge: str = "gasteiger",
    optimize: bool = True,
    show_progress: bool = False,
    ligand_id: str = None,
) -> bool:
    """Convert SMILES to PDBQT (complete pipeline)."""

    # Setup progress bar if requested
    pbar = None
    if show_progress:
        ligand_label = ligand_id or pdbqt_path.stem
        steps = ['Parsing', '3D Gen', 'Optimize', 'Convert']
        pbar = tqdm(total=len(steps), desc=f"Preparing {ligand_label}", unit="step", ncols=80)

    def update_progress(step_idx):
        if pbar:
            pbar.set_postfix_str(steps[step_idx])
            pbar.update(1)

    try:
        # Generate 3D structure
        update_progress(0)
        update_progress(1)
        mol = smiles_to_mol_with_3d(smiles=smiles, optimize=optimize)
        update_progress(2)

        if mol is None:
            return False

        # Write to temporary SDF and convert
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp:
            tmp_sdf = Path(tmp.name)

        try:
            writer = Chem.SDWriter(str(tmp_sdf))
            writer.write(mol)
            writer.close()

            update_progress(3)
            success = sdf_to_pdbqt(
                sdf_path=tmp_sdf,
                pdbqt_path=pdbqt_path,
                ph=ph,
                partial_charge=partial_charge,
            )

            return success
        finally:
            if tmp_sdf.exists():
                tmp_sdf.unlink()
    finally:
        if pbar:
            pbar.close()


def main():
    parser = argparse.ArgumentParser(description="Convert SMILES to PDBQT format for docking")
    parser.add_argument("--smiles", type=str, required=True, help="SMILES string")
    parser.add_argument("--output", type=Path, required=True, help="Output PDBQT file")
    parser.add_argument("--ligand-id", type=str, help="Ligand identifier (for logging)")
    parser.add_argument("--ph", type=float, default=7.4, help="pH for protonation (default: 7.4)")
    parser.add_argument("--partial-charge", type=str, default="gasteiger",
                        choices=["gasteiger", "mmff94", "eem"], help="Partial charge method")
    parser.add_argument("--no-optimize", action="store_true", help="Skip geometry optimization")
    parser.add_argument("--progress", action="store_true", help="Show progress bar")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    parser.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        print(f"ERROR: Output file already exists: {args.output}", file=sys.stderr)
        print(f"Use --overwrite to replace it, or specify a different output path.", file=sys.stderr)
        sys.exit(1)

    ligand_label = args.ligand_id or args.output.stem
    if not args.progress and not args.quiet:
        print(f"Processing: {ligand_label}")
        print(f"  SMILES: {args.smiles}")

    success = smiles_to_pdbqt(
        smiles=args.smiles,
        pdbqt_path=args.output,
        ph=args.ph,
        partial_charge=args.partial_charge,
        optimize=not args.no_optimize,
        show_progress=args.progress,
        ligand_id=args.ligand_id,
    )

    if success:
        if not args.quiet:
            print(f"✓ Created: {args.output}")
        sys.exit(0)
    else:
        print(f"✗ Failed to process: {ligand_label}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
