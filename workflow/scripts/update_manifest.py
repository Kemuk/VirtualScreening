#!/usr/bin/env python3
"""
update_manifest.py

Update specific columns in the manifest for given compound keys.
Used by individual workflow stages to mark completion and store results.

Examples:
  # Mark preparation complete
  update_manifest.py --keys ADRB2_compound_001 --set preparation_status=True

  # Update docking results
  update_manifest.py --keys ADRB2_compound_001 --set docking_status=True vina_score=-8.5

  # Bulk update from file
  update_manifest.py --keys-file completed.txt --set rescoring_status=True
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import pyarrow.parquet as pq
import pandas as pd


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load manifest from Parquet file."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    table = pq.read_table(manifest_path)
    return table.to_pandas()


def save_manifest(df: pd.DataFrame, manifest_path: Path, backup: bool = True):
    """Save updated manifest, optionally creating a backup."""
    if backup and manifest_path.exists():
        backup_dir = manifest_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"manifest_{timestamp}.parquet"
        manifest_path.rename(backup_path)

    # Ensure timestamps are datetime
    df['last_updated'] = pd.to_datetime(df['last_updated'])
    df['created_at'] = pd.to_datetime(df['created_at'])

    table = pq.Table.from_pandas(df)
    pq.write_table(table, manifest_path, compression='snappy')


def parse_updates(update_args: List[str]) -> Dict[str, Any]:
    """
    Parse update arguments in format: column=value

    Handles type conversion:
      - True/False -> bool
      - numeric -> float
      - else -> string
    """
    updates = {}

    for arg in update_args:
        if '=' not in arg:
            raise ValueError(f"Invalid update format: {arg}. Expected: column=value")

        column, value_str = arg.split('=', 1)

        # Type conversion
        if value_str.lower() == 'true':
            value = True
        elif value_str.lower() == 'false':
            value = False
        elif value_str.lower() in ('none', 'null', ''):
            value = None
        else:
            try:
                value = float(value_str)
            except ValueError:
                value = value_str

        updates[column] = value

    return updates


def update_manifest_entries(
    df: pd.DataFrame,
    compound_keys: List[str],
    updates: Dict[str, Any],
) -> pd.DataFrame:
    """
    Update manifest entries for specified compound keys.

    Returns updated DataFrame.
    """
    # Filter to rows that need updating
    mask = df['compound_key'].isin(compound_keys)
    num_matches = mask.sum()

    if num_matches == 0:
        print(f"WARNING: No matches found for {len(compound_keys)} compound keys", file=sys.stderr)
        return df

    # Apply updates
    for column, value in updates.items():
        if column not in df.columns:
            raise ValueError(f"Unknown column: {column}")

        df.loc[mask, column] = value

    # Update last_updated timestamp
    df.loc[mask, 'last_updated'] = datetime.now()

    print(f"Updated {num_matches} entries")
    for column, value in updates.items():
        print(f"  {column} = {value}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Update manifest entries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update single compound
  %(prog)s --manifest data/master/manifest.parquet \\
           --keys ADRB2_compound_001 \\
           --set preparation_status=True

  # Update multiple compounds
  %(prog)s --keys ADRB2_001 ADRB2_002 ALDH1_003 \\
           --set docking_status=True vina_score=-8.5

  # Update from file (one compound_key per line)
  %(prog)s --keys-file completed.txt \\
           --set rescoring_status=True aev_plig_score=0.85
"""
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/master/manifest.parquet"),
        help="Path to manifest file"
    )
    parser.add_argument(
        "--keys",
        nargs='+',
        help="Compound keys to update"
    )
    parser.add_argument(
        "--keys-file",
        type=Path,
        help="File containing compound keys (one per line)"
    )
    parser.add_argument(
        "--set",
        nargs='+',
        required=True,
        help="Updates in format: column=value"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup before updating"
    )

    args = parser.parse_args()

    # Collect compound keys
    compound_keys = []

    if args.keys:
        compound_keys.extend(args.keys)

    if args.keys_file:
        with open(args.keys_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    compound_keys.append(line)

    if not compound_keys:
        print("ERROR: No compound keys specified", file=sys.stderr)
        sys.exit(1)

    # Parse updates
    try:
        updates = parse_updates(args.set)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Load manifest
    print(f"Loading manifest: {args.manifest}")
    df = load_manifest(args.manifest)
    print(f"  Total entries: {len(df)}")

    # Apply updates
    df_updated = update_manifest_entries(df, compound_keys, updates)

    # Save manifest
    save_manifest(df_updated, args.manifest, backup=not args.no_backup)
    print(f"✓ Manifest updated successfully")


if __name__ == "__main__":
    main()
