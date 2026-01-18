#!/usr/bin/env python3
"""
mol2_to_pdbqt.py

Convert receptor MOL2 file to PDBQT and PDB formats.

Uses OpenBabel for format conversion with proper hydrogen handling
and charge assignment.

Output:
  - receptor.pdbqt (for Vina docking)
  - receptor.pdb (for visualization and AEV-PLIG)
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def convert_mol2_to_pdbqt(
    mol2_path: Path,
    pdbqt_path: Path,
    ph: float = 7.4,
    partial_charge: str = "gasteiger",
) -> bool:
    """
    Convert MOL2 to PDBQT using OpenBabel.

    Creates a RIGID receptor (no flexibility) for Vina docking.

    Args:
        mol2_path: Input MOL2 file
        pdbqt_path: Output PDBQT file
        ph: pH for hydrogen addition
        partial_charge: Charge calculation method

    Returns:
        True if successful, False otherwise
    """
    mol2_path = mol2_path.expanduser().resolve()
    pdbqt_path = pdbqt_path.expanduser().resolve()
    pdbqt_path.parent.mkdir(parents=True, exist_ok=True)
    obabel_bin = os.environ.get("OBABEL_BIN", "obabel")

    # OpenBabel command for MOL2 → PDBQT conversion (RIGID receptor)
    # -xr: Make receptor rigid (no TORSDOF/BRANCH records)
    # -p: Add hydrogens at pH
    # --partialcharge: Calculate partial charges
    cmd = [
        obabel_bin,
        str(mol2_path),
        "-O", str(pdbqt_path),
        "-xr",  # Rigid receptor flag - critical for Vina!
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
        print(f"✓ Created PDBQT: {pdbqt_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR: OpenBabel conversion failed", file=sys.stderr)
        print(f"Command: {' '.join(cmd)}", file=sys.stderr)
        print(f"Exit code: {e.returncode}", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)
        return False

    except FileNotFoundError:
        print("ERROR: obabel not found. Install OpenBabel.", file=sys.stderr)
        return False


def convert_mol2_to_pdb(
    mol2_path: Path,
    pdb_path: Path,
) -> bool:
    """
    Convert MOL2 to PDB using OpenBabel.

    Args:
        mol2_path: Input MOL2 file
        pdb_path: Output PDB file

    Returns:
        True if successful, False otherwise
    """
    mol2_path = mol2_path.expanduser().resolve()
    pdb_path = pdb_path.expanduser().resolve()
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    obabel_bin = os.environ.get("OBABEL_BIN", "obabel")

    cmd = [
        obabel_bin,
        str(mol2_path),
        "-O", str(pdb_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"✓ Created PDB: {pdb_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR: OpenBabel conversion failed", file=sys.stderr)
        print(f"Command: {' '.join(cmd)}", file=sys.stderr)
        print(f"Exit code: {e.returncode}", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)
        return False

    except FileNotFoundError:
        print("ERROR: obabel not found. Install OpenBabel.", file=sys.stderr)
        return False


# =============================================================================
# Batch Processing (for SLURM array jobs)
# =============================================================================

def process_batch(items: list, config: dict) -> list:
    """
    Process a batch of receptor conversions.

    Called by the SLURM worker to process a chunk of items.
    For receptors, each item represents a unique protein target.

    Args:
        items: List of item records from manifest (dicts with protein info)
        config: Workflow configuration dict

    Returns:
        List of result records with 'ligand_id', 'success', 'error'
    """
    import yaml

    results = []
    prep_config = config.get('preparation', {})
    ph = prep_config.get('ph', 7.4)
    partial_charge = prep_config.get('partial_charge', 'gasteiger')

    # Load targets config for receptor paths
    targets_path = Path(config.get('targets_config', 'config/targets.yaml'))
    with open(targets_path) as f:
        targets_config = yaml.safe_load(f)

    # Track which proteins we've already processed
    processed_proteins = set()

    for item in items:
        ligand_id = item['ligand_id']
        protein_id = item['protein_id']

        # Skip if already processed this protein
        if protein_id in processed_proteins:
            results.append({
                'ligand_id': ligand_id,
                'success': True,
                'skipped': True,
            })
            continue

        processed_proteins.add(protein_id)

        try:
            # Get receptor paths from targets config
            target_cfg = targets_config['targets'].get(protein_id)
            if not target_cfg:
                results.append({
                    'ligand_id': ligand_id,
                    'success': False,
                    'error': f'Target {protein_id} not found in targets.yaml',
                })
                continue

            mol2_path = Path(target_cfg['receptor_mol2'])
            pdbqt_path = Path(item['receptor_pdbqt_path'])
            pdb_path = Path(item['receptor_pdb_path'])

            # Skip if already converted
            if pdbqt_path.exists() and pdb_path.exists():
                results.append({
                    'ligand_id': ligand_id,
                    'success': True,
                    'skipped': True,
                })
                continue

            # Convert to PDBQT
            success_pdbqt = convert_mol2_to_pdbqt(
                mol2_path=mol2_path,
                pdbqt_path=pdbqt_path,
                ph=ph,
                partial_charge=partial_charge,
            )

            # Convert to PDB
            success_pdb = convert_mol2_to_pdb(
                mol2_path=mol2_path,
                pdb_path=pdb_path,
            )

            results.append({
                'ligand_id': ligand_id,
                'success': success_pdbqt and success_pdb,
            })

        except Exception as e:
            results.append({
                'ligand_id': ligand_id,
                'success': False,
                'error': str(e),
            })

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert receptor MOL2 to PDBQT and PDB formats"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input MOL2 file"
    )
    parser.add_argument(
        "--pdbqt",
        type=Path,
        required=True,
        help="Output PDBQT file"
    )
    parser.add_argument(
        "--pdb",
        type=Path,
        required=True,
        help="Output PDB file"
    )
    parser.add_argument(
        "--ph",
        type=float,
        default=7.4,
        help="pH for hydrogen addition (default: 7.4)"
    )
    parser.add_argument(
        "--partial-charge",
        type=str,
        default="gasteiger",
        choices=["gasteiger", "mmff94", "eem"],
        help="Partial charge calculation method (default: gasteiger)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files (required if outputs exist)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Check if outputs exist and overwrite flag is needed
    if args.pdbqt.exists() or args.pdb.exists():
        if not args.overwrite:
            print(f"ERROR: Output files already exist:", file=sys.stderr)
            if args.pdbqt.exists():
                print(f"  PDBQT: {args.pdbqt}", file=sys.stderr)
            if args.pdb.exists():
                print(f"  PDB: {args.pdb}", file=sys.stderr)
            print(f"Use --overwrite to replace them, or specify different output paths.", file=sys.stderr)
            sys.exit(1)

    print(f"Converting receptor: {args.input}")
    print(f"  pH: {args.ph}")
    print(f"  Partial charge method: {args.partial_charge}")

    # Convert to PDBQT
    success_pdbqt = convert_mol2_to_pdbqt(
        mol2_path=args.input,
        pdbqt_path=args.pdbqt,
        ph=args.ph,
        partial_charge=args.partial_charge,
    )

    # Convert to PDB
    success_pdb = convert_mol2_to_pdb(
        mol2_path=args.input,
        pdb_path=args.pdb,
    )

    if success_pdbqt and success_pdb:
        print("✓ Receptor preparation complete!")
        sys.exit(0)
    else:
        print("✗ Receptor preparation failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
