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

This script processes a single ligand at a time.
Batch processing is handled by Snakemake parallel execution.
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
    """
    Convert SMILES to RDKit Mol with 3D coordinates.

    Args:
        smiles: SMILES string
        optimize: Whether to optimize geometry with MMFF
        max_iters: Maximum optimization iterations

    Returns:
        RDKit Mol object with 3D coordinates, or None if failed
    """
    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"ERROR: Invalid SMILES: {smiles}", file=sys.stderr)
        return None

    # Add hydrogens
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    try:
        result = AllChem.EmbedMolecule(
            mol,
            randomSeed=42,
            useRandomCoords=False,
        )

        if result != 0:
            print(f"ERROR: 3D coordinate generation failed (code: {result})", file=sys.stderr)
            return None

    except Exception as e:
        print(f"ERROR: 3D embedding failed: {e}", file=sys.stderr)
        return None

    # Optimize geometry with MMFF
    if optimize:
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
        except Exception as e:
            print(f"WARNING: MMFF optimization failed: {e}", file=sys.stderr)
            # Continue anyway - non-optimized geometry is better than failure

    return mol


def mol_to_sdf(mol: Chem.Mol, sdf_path: Path) -> bool:
    """
    Write RDKit Mol to SDF file.

    Args:
        mol: RDKit Mol object
        sdf_path: Output SDF path

    Returns:
        True if successful
    """
    try:
        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()
        return True
    except Exception as e:
        print(f"ERROR: Failed to write SDF: {e}", file=sys.stderr)
        return False


def sdf_to_pdbqt(
    sdf_path: Path,
    pdbqt_path: Path,
    ph: float = 7.4,
    partial_charge: str = "gasteiger",
) -> bool:
    """
    Convert SDF to PDBQT using OpenBabel.

    Args:
        sdf_path: Input SDF file
        pdbqt_path: Output PDBQT file
        ph: pH for protonation
        partial_charge: Charge calculation method

    Returns:
        True if successful
    """
    pdbqt_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "obabel",
        str(sdf_path),
        "-O", str(pdbqt_path),
        "-p", str(ph),
        "--partialcharge", partial_charge,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
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
    gen3d: bool = True,
    optimize: bool = True,
    show_progress: bool = False,
    ligand_id: str = None,
) -> bool:
    """
    Convert SMILES to PDBQT (complete pipeline).

    Args:
        smiles: SMILES string
        pdbqt_path: Output PDBQT path
        ph: pH for protonation
        partial_charge: Charge calculation method
        gen3d: Generate 3D coordinates
        optimize: Optimize geometry
        show_progress: Show progress bar
        ligand_id: Ligand identifier for progress display

    Returns:
        True if successful
    """
    if show_progress:
        ligand_label = ligand_id or pdbqt_path.stem
        steps = ['Parsing SMILES', 'Generating 3D', 'Optimizing', 'Converting to PDBQT']
        with tqdm(total=len(steps), desc=f"Preparing {ligand_label}", unit="step", ncols=80) as pbar:
            # Generate 3D structure
            pbar.set_postfix_str(steps[0])
            pbar.update(1)

            pbar.set_postfix_str(steps[1])
            mol = smiles_to_mol_with_3d(
                smiles=smiles,
                optimize=optimize,
            )
            pbar.update(1)

            if mol is None:
                return False

            # Write to temporary SDF
            with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp:
                tmp_sdf = Path(tmp.name)

            try:
                pbar.set_postfix_str(steps[2])
                if not mol_to_sdf(mol, tmp_sdf):
                    return False
                pbar.update(1)

                # Convert SDF to PDBQT
                pbar.set_postfix_str(steps[3])
                success = sdf_to_pdbqt(
                    sdf_path=tmp_sdf,
                    pdbqt_path=pdbqt_path,
                    ph=ph,
                    partial_charge=partial_charge,
                )
                pbar.update(1)

                return success

            finally:
                # Clean up temporary file
                if tmp_sdf.exists():
                    tmp_sdf.unlink()
    else:
        # Generate 3D structure (no progress)
        mol = smiles_to_mol_with_3d(
            smiles=smiles,
            optimize=optimize,
        )

        if mol is None:
            return False

        # Write to temporary SDF
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp:
            tmp_sdf = Path(tmp.name)

        try:
            if not mol_to_sdf(mol, tmp_sdf):
                return False

            # Convert SDF to PDBQT
            success = sdf_to_pdbqt(
                sdf_path=tmp_sdf,
                pdbqt_path=pdbqt_path,
                ph=ph,
                partial_charge=partial_charge,
            )

            return success

        finally:
            # Clean up temporary file
            if tmp_sdf.exists():
                tmp_sdf.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Convert SMILES to PDBQT format for docking"
    )
    parser.add_argument(
        "--smiles",
        type=str,
        required=True,
        help="SMILES string"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PDBQT file"
    )
    parser.add_argument(
        "--ligand-id",
        type=str,
        help="Ligand identifier (for logging)"
    )
    parser.add_argument(
        "--ph",
        type=float,
        default=7.4,
        help="pH for protonation (default: 7.4)"
    )
    parser.add_argument(
        "--partial-charge",
        type=str,
        default="gasteiger",
        choices=["gasteiger", "mmff94", "eem"],
        help="Partial charge method (default: gasteiger)"
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip geometry optimization"
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bar during preparation"
    )

    args = parser.parse_args()

    ligand_label = args.ligand_id or args.output.stem
    if not args.progress:
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
        print(f"✓ Created: {args.output}")
        sys.exit(0)
    else:
        print(f"✗ Failed to process: {ligand_label}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
