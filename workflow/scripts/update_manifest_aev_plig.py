#!/usr/bin/env python3
"""
update_manifest_aev_plig.py

Update manifest with AEV-PLIG prediction results.

Reads predictions from AEV-PLIG output CSV and updates the manifest parquet with:
    - binding_affinity_pK: converted from Vina score
    - aev_plig_best_score: ensemble prediction (preds column)
    - aev_prediction_0-9: individual model predictions
    - rescoring_status: set to True for processed ligands
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import pyarrow as pa
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


def update_manifest_with_predictions(
    manifest_path: Path,
    predictions_path: Path,
    backup: bool = True,
) -> int:
    """
    Update manifest with AEV-PLIG predictions.

    Args:
        manifest_path: Path to manifest parquet file
        predictions_path: Path to AEV-PLIG predictions CSV
        backup: Whether to create backup of manifest

    Returns:
        Number of ligands updated
    """
    # Load manifest
    print(f"Loading manifest from {manifest_path}...")
    manifest = pq.read_table(manifest_path).to_pandas()
    print(f"  Total entries: {len(manifest)}")

    # Load predictions
    print(f"Loading predictions from {predictions_path}...")
    predictions = pd.read_csv(predictions_path)
    print(f"  Total predictions: {len(predictions)}")

    # Validate predictions columns
    required_cols = ['unique_id', 'preds']
    pred_cols = [f'preds_{i}' for i in range(10)]
    required_cols.extend(pred_cols)

    missing_cols = [c for c in required_cols if c not in predictions.columns]
    if missing_cols:
        print(f"ERROR: Missing columns in predictions: {missing_cols}", file=sys.stderr)
        sys.exit(1)

    # Create lookup dict from predictions (unique_id -> row)
    pred_lookup = predictions.set_index('unique_id').to_dict('index')
    print(f"  Unique predictions: {len(pred_lookup)}")

    # Update manifest
    updated_count = 0
    now = datetime.now()

    for idx, row in manifest.iterrows():
        compound_key = row['compound_key']

        if compound_key in pred_lookup:
            pred_row = pred_lookup[compound_key]

            # Update binding_affinity_pK from vina_score
            if pd.notna(row['vina_score']):
                manifest.at[idx, 'binding_affinity_pK'] = vina_score_to_pK(row['vina_score'])

            # Update AEV-PLIG predictions
            manifest.at[idx, 'aev_plig_best_score'] = pred_row['preds']

            for i in range(10):
                manifest.at[idx, f'aev_prediction_{i}'] = pred_row[f'preds_{i}']

            # Mark as rescored
            manifest.at[idx, 'rescoring_status'] = True
            manifest.at[idx, 'last_updated'] = now

            updated_count += 1

    print(f"\nUpdated {updated_count} ligands in manifest")

    # Create backup if requested
    if backup and manifest_path.exists():
        backup_dir = manifest_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"manifest_pre_aev_plig_{timestamp}.parquet"

        # Copy current file to backup
        import shutil
        shutil.copy(manifest_path, backup_path)
        print(f"Created backup: {backup_path}")

    # Save updated manifest
    # Re-read schema from original file to preserve it
    original_schema = pq.read_schema(manifest_path)

    # Write updated manifest
    table = pa.Table.from_pandas(manifest, schema=original_schema, preserve_index=False)
    pq.write_table(table, manifest_path, compression='snappy')

    print(f"Saved updated manifest: {manifest_path}")

    # Print summary statistics
    rescored = manifest[manifest['rescoring_status'] == True]
    print(f"\nRescoring summary:")
    print(f"  Total rescored: {len(rescored)}")
    if len(rescored) > 0:
        print(f"  AEV-PLIG best score range: {rescored['aev_plig_best_score'].min():.2f} - {rescored['aev_plig_best_score'].max():.2f}")
        print(f"  AEV-PLIG best score mean: {rescored['aev_plig_best_score'].mean():.2f}")

    return updated_count


def main():
    parser = argparse.ArgumentParser(
        description="Update manifest with AEV-PLIG predictions"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest parquet file"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to AEV-PLIG predictions CSV"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup of manifest before updating"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    if not args.predictions.exists():
        print(f"ERROR: Predictions file not found: {args.predictions}", file=sys.stderr)
        sys.exit(1)

    # Update manifest
    updated_count = update_manifest_with_predictions(
        manifest_path=args.manifest,
        predictions_path=args.predictions,
        backup=not args.no_backup,
    )

    if updated_count == 0:
        print("WARNING: No ligands were updated", file=sys.stderr)
        sys.exit(1)

    print(f"\nManifest update complete!")
    sys.exit(0)


if __name__ == "__main__":
    main()
