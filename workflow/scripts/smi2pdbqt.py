#!/usr/bin/env python3
"""
smi2pdbqt.py

Convert SMILES to PDBQT format using OpenBabel directly.
"""

import argparse
import sys
import subprocess
from pathlib import Path


def smiles_to_pdbqt(
    smiles: str,
    pdbqt_path: Path,
    ph: float = 7.4,
    partial_charge: str = "gasteiger",
) -> bool:
    """
    Convert SMILES to PDBQT using OpenBabel directly.

    Args:
        smiles: SMILES string
        pdbqt_path: Output PDBQT path
        ph: pH for protonation
        partial_charge: Charge calculation method

    Returns:
        True if successful
    """
    pdbqt_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "obabel",
        f"-:{smiles}",
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


def main():
    parser = argparse.ArgumentParser(description="Convert SMILES to PDBQT format")
    parser.add_argument("--smiles", type=str, required=True, help="SMILES string")
    parser.add_argument("--output", type=Path, required=True, help="Output PDBQT file")
    parser.add_argument("--ligand-id", type=str, help="Ligand identifier (for logging)")
    parser.add_argument("--ph", type=float, default=7.4, help="pH for protonation (default: 7.4)")
    parser.add_argument("--partial-charge", type=str, default="gasteiger",
                        choices=["gasteiger", "mmff94", "eem"], help="Partial charge method")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    parser.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        print(f"ERROR: Output file already exists: {args.output}", file=sys.stderr)
        print(f"Use --overwrite to replace it, or specify a different output path.", file=sys.stderr)
        sys.exit(1)

    ligand_label = args.ligand_id or args.output.stem
    if not args.quiet:
        print(f"Processing: {ligand_label}")
        print(f"  SMILES: {args.smiles}")

    success = smiles_to_pdbqt(
        smiles=args.smiles,
        pdbqt_path=args.output,
        ph=args.ph,
        partial_charge=args.partial_charge,
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
