#!/usr/bin/env python3
"""
prepare_aev_plig_csv.py

Generate input CSV for AEV-PLIG rescoring from the manifest.

Creates a CSV file with format:
    unique_id,pK,sdf_file,pdb_file

Where:
    - unique_id: compound_key from manifest
    - pK: binding affinity converted from Vina score
    - sdf_file: path to docked SDF file
    - pdb_file: path to receptor PDB file
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import pyarrow.parquet as pq


def vina_score_to_pK(vina_score: float, temperature: float = 298.0) -> float:
    """
    Convert Vina docking score to pK using thermodynamic relationship.

    pK = -ΔG / (2.303 * R * T)

    Args:
        vina_score: Vina affinity in kcal/mol (negative values = favorable)
        temperature: Temperature in Kelvin (default: 298K)

    Returns:
        pK value (higher = stronger binding)
    """
    R = 0.001987  # kcal/(mol·K)
    return -vina_score / (2.303 * R * temperature)


def prepare_aev_plig_csv(
    manifest_path: Path,
    output_path: Path,
    project_root: Optional[Path] = None,
) -> int:
    """
    Generate AEV-PLIG input CSV from manifest.

    Args:
        manifest_path: Path to manifest parquet file
        output_path: Output CSV path
        project_root: Project root for resolving relative paths

    Returns:
        Number of ligands in output CSV
    """
    if project_root is None:
        project_root = Path.cwd()

    # Load manifest
    print(f"Loading manifest from {manifest_path}...")
    manifest = pq.read_table(manifest_path).to_pandas()
    print(f"  Total entries: {len(manifest)}")

    # Filter to docked ligands with valid SDF files
    docked = manifest[manifest['docking_status'] == True].copy()
    print(f"  Docked ligands: {len(docked)}")

    if len(docked) == 0:
        print("ERROR: No docked ligands found in manifest", file=sys.stderr)
        return 0

    # Check for valid SDF files
    def sdf_exists(sdf_path: str) -> bool:
        if pd.isna(sdf_path):
            return False
        full_path = project_root / sdf_path
        return full_path.exists()

    docked['sdf_exists'] = docked['docked_sdf_path'].apply(sdf_exists)
    valid = docked[docked['sdf_exists']].copy()
    print(f"  With valid SDF files: {len(valid)}")

    if len(valid) == 0:
        print("ERROR: No ligands with valid SDF files found", file=sys.stderr)
        return 0

    # Check for valid Vina scores
    valid = valid[valid['vina_score'].notna()].copy()
    print(f"  With valid Vina scores: {len(valid)}")

    if len(valid) == 0:
        print("ERROR: No ligands with valid Vina scores found", file=sys.stderr)
        return 0

    # Compute pK values
    valid['pK'] = valid['vina_score'].apply(vina_score_to_pK)

    # Build output dataframe
    output_rows = []
    for _, row in valid.iterrows():
        # Resolve paths to absolute
        sdf_path = project_root / row['docked_sdf_path']
        pdb_path = project_root / row['receptor_pdb_path']

        output_rows.append({
            'unique_id': row['compound_key'],
            'pK': row['pK'],
            'sdf_file': str(sdf_path.resolve()),
            'pdb_file': str(pdb_path.resolve()),
        })

    output_df = pd.DataFrame(output_rows)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    output_df.to_csv(output_path, index=False)

    print(f"\nCreated AEV-PLIG input CSV: {output_path}")
    print(f"  Ligands: {len(output_df)}")
    print(f"  pK range: {output_df['pK'].min():.2f} - {output_df['pK'].max():.2f}")

    return len(output_df)


def main():
    parser = argparse.ArgumentParser(
        description="Generate AEV-PLIG input CSV from manifest"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest parquet file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("AEV-PLIG/data/lit_pcba.csv"),
        help="Output CSV path (default: AEV-PLIG/data/lit_pcba.csv)"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory for resolving paths"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    # Generate CSV
    num_ligands = prepare_aev_plig_csv(
        manifest_path=args.manifest,
        output_path=args.output,
        project_root=args.project_root,
    )

    if num_ligands == 0:
        print("ERROR: No ligands processed", file=sys.stderr)
        sys.exit(1)

    print(f"\nAEV-PLIG input preparation complete!")
    sys.exit(0)


if __name__ == "__main__":
    main()
