#!/usr/bin/env python3
"""
rescore_aev_plig.py

Prepare data for AEV-PLIG rescoring by combining molecular properties
with docking scores.

Creates a CSV file with:
  - Molecular descriptors (MW, LogP, HBD, HBA)
  - Docking scores (Vina affinity, pK)
  - File paths (SDF, protein PDB)
  - Activity labels

This CSV is used as input for the AEV-PLIG neural network rescoring.
"""

import argparse
import sys
import csv
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit import rdBase
import warnings

# Suppress RDKit warnings
rdBase.DisableLog('rdApp.warning')
warnings.filterwarnings("ignore", category=UserWarning)


# Regex to match Vina affinity in PDBQT REMARK lines
REMARK_VINA_RE = re.compile(
    r"REMARK.*VINA(?:\s+RESULT)?[:\s]+\s*(-?\d+\.\d+)",
    re.IGNORECASE
)


def compute_rdkit_properties(sdf_path: Path) -> Optional[Tuple[float, float, int, int]]:
    """
    Compute RDKit molecular descriptors from SDF file.

    Args:
        sdf_path: Path to SDF file

    Returns:
        Tuple of (MW, LogP, HBD, HBA) or None if failed
    """
    try:
        suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
        mol = next((m for m in suppl if m is not None), None)

        if mol is None:
            return None

        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)

        return (mw, logp, hbd, hba)

    except Exception as e:
        print(f"WARNING: Failed to compute properties for {sdf_path}: {e}", file=sys.stderr)
        return None


def parse_vina_affinity(pdbqt_path: Path) -> Optional[float]:
    """
    Extract Vina binding affinity from docked PDBQT file.

    Searches for REMARK lines like:
      REMARK VINA RESULT:    -8.5      0.000      0.000

    Args:
        pdbqt_path: Path to docked PDBQT file

    Returns:
        Binding affinity (kcal/mol) or None if not found
    """
    if not pdbqt_path.exists():
        return None

    try:
        with open(pdbqt_path) as f:
            for line in f:
                match = REMARK_VINA_RE.search(line)
                if match:
                    return float(match.group(1))
    except Exception as e:
        print(f"WARNING: Failed to parse affinity from {pdbqt_path}: {e}", file=sys.stderr)
        return None

    return None


def docking_score_to_pK(docking_score: float, temperature: float = 298.0) -> float:
    """
    Convert Vina docking score to pK using thermodynamic relationship.

    pK = -ΔG / (2.303 * R * T)

    Args:
        docking_score: Vina affinity in kcal/mol
        temperature: Temperature in Kelvin (default: 298K)

    Returns:
        pK value
    """
    R = 0.001987  # kcal/(mol·K)
    return -docking_score / (2.303 * R * temperature)


def find_protein_pdb(receptor_pdbqt: Path) -> Optional[Path]:
    """
    Find protein PDB file corresponding to receptor PDBQT.

    Looks for:
      1. Same name with .pdb extension (e.g., receptor.pdb)
      2. Any PDB file in the same directory

    Args:
        receptor_pdbqt: Path to receptor PDBQT file

    Returns:
        Path to protein PDB or None
    """
    if not receptor_pdbqt.exists():
        return None

    # Try same name with .pdb extension
    pdb_path = receptor_pdbqt.with_suffix('.pdb')
    if pdb_path.exists():
        return pdb_path

    # Try any PDB in same directory
    pdb_files = list(receptor_pdbqt.parent.glob('*.pdb'))
    if pdb_files:
        return pdb_files[0]

    return None


def process_ligand_for_rescoring(
    ligand_id: str,
    protein_id: str,
    sdf_path: Path,
    pdbqt_path: Path,
    receptor_pdbqt: Path,
    is_active: bool,
) -> Optional[Dict]:
    """
    Process a single ligand and extract all data for AEV-PLIG.

    Args:
        ligand_id: Ligand identifier
        protein_id: Protein target ID
        sdf_path: Path to docked SDF file
        pdbqt_path: Path to docked PDBQT file
        receptor_pdbqt: Path to receptor PDBQT
        is_active: Whether ligand is active

    Returns:
        Dictionary with all rescoring data or None if failed
    """
    # Compute molecular properties
    props = compute_rdkit_properties(sdf_path)
    if props is None:
        print(f"WARNING: Skipping {ligand_id} - failed to compute properties", file=sys.stderr)
        return None

    mw, logp, hbd, hba = props

    # Parse Vina affinity
    docking_score = parse_vina_affinity(pdbqt_path)
    if docking_score is None:
        print(f"WARNING: Skipping {ligand_id} - no Vina affinity found", file=sys.stderr)
        return None

    # Convert to pK
    pK = docking_score_to_pK(docking_score)

    # Find protein PDB
    protein_pdb = find_protein_pdb(receptor_pdbqt)
    if protein_pdb is None:
        print(f"WARNING: No protein PDB found for {protein_id}", file=sys.stderr)
        protein_pdb_str = ""
    else:
        protein_pdb_str = str(protein_pdb.resolve())

    # Create unique ID
    unique_id = f"{protein_id}_{ligand_id}"

    return {
        'unique_id': unique_id,
        'Protein_ID': protein_id,
        'sdf_file': str(sdf_path.resolve()),
        'protein_pdb': protein_pdb_str,
        'MW': mw,
        'LogP': logp,
        'HBD': hbd,
        'HBA': hba,
        'DockingScore': docking_score,
        'pK': pK,
        'is_active': int(is_active),
    }


def process_manifest_for_target(
    manifest_path: Path,
    target_id: str,
    output_csv: Path,
) -> int:
    """
    Process all docked ligands for a target and create AEV-PLIG CSV.

    Args:
        manifest_path: Path to manifest Parquet file
        target_id: Target protein ID
        output_csv: Output CSV path

    Returns:
        Number of ligands processed
    """
    import pyarrow.parquet as pq

    # Load manifest
    manifest = pq.read_table(manifest_path).to_pandas()

    # Filter to this target and docked ligands
    target_ligands = manifest[
        (manifest['protein_id'] == target_id) &
        (manifest['docking_status'] == True)
    ]

    if len(target_ligands) == 0:
        print(f"WARNING: No docked ligands found for {target_id}", file=sys.stderr)
        return 0

    print(f"Processing {len(target_ligands)} docked ligands for {target_id}")

    # Process each ligand
    results = []
    for _, row in target_ligands.iterrows():
        result = process_ligand_for_rescoring(
            ligand_id=row['ligand_id'],
            protein_id=row['protein_id'],
            sdf_path=Path(row['docked_sdf_path']),
            pdbqt_path=Path(row['docked_pdbqt_path']),
            receptor_pdbqt=Path(row['receptor_pdbqt_path']),
            is_active=row['is_active'],
        )

        if result is not None:
            results.append(result)

    if not results:
        print(f"ERROR: No ligands successfully processed for {target_id}", file=sys.stderr)
        return 0

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)

    # Ensure column order
    columns = [
        'unique_id', 'Protein_ID', 'sdf_file', 'protein_pdb',
        'MW', 'LogP', 'HBD', 'HBA', 'DockingScore', 'pK', 'is_active'
    ]
    df = df[columns]

    df.to_csv(output_csv, index=False)

    print(f"✓ Created AEV-PLIG CSV: {output_csv}")
    print(f"  Ligands: {len(results)}")
    print(f"  Actives: {df['is_active'].sum()}")
    print(f"  Inactives: {(~df['is_active'].astype(bool)).sum()}")

    return len(results)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare AEV-PLIG rescoring data from manifest"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest Parquet file"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target protein ID"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file for AEV-PLIG"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    # Process ligands
    num_processed = process_manifest_for_target(
        manifest_path=args.manifest,
        target_id=args.target,
        output_csv=args.output,
    )

    if num_processed == 0:
        print(f"ERROR: No ligands processed for {args.target}", file=sys.stderr)
        sys.exit(1)

    print(f"✓ AEV-PLIG data preparation complete!")
    sys.exit(0)


if __name__ == "__main__":
    main()
