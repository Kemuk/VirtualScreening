#!/usr/bin/env python3
"""
deduplicate_manifest.py

Remove duplicate rows from manifest, keeping only unique compound_keys.

Usage:
    python -m workflow.slurm.deduplicate_manifest
    python -m workflow.slurm.deduplicate_manifest --keep first  # keep first occurrence
    python -m workflow.slurm.deduplicate_manifest --keep last   # keep last occurrence
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def deduplicate_manifest(
    manifest_path: Path,
    keep: str = 'first',
    dry_run: bool = False,
) -> int:
    """
    Remove duplicate compound_keys from manifest.

    Args:
        manifest_path: Path to manifest.parquet
        keep: Which duplicate to keep ('first' or 'last')
        dry_run: If True, don't write changes

    Returns:
        Number of duplicates removed
    """
    # Load manifest
    df = pq.read_table(manifest_path).to_pandas()
    original_count = len(df)

    print(f"Loaded manifest: {original_count:,} rows")

    # Check for duplicates
    duplicates = df[df.duplicated(subset='compound_key', keep=False)]
    num_duplicated_keys = duplicates['compound_key'].nunique()
    num_duplicate_rows = len(duplicates) - num_duplicated_keys

    if num_duplicate_rows == 0:
        print("No duplicates found.")
        return 0

    print(f"\nFound {num_duplicated_keys:,} compound_keys with duplicates")
    print(f"Total duplicate rows to remove: {num_duplicate_rows:,}")

    # Show sample duplicates
    print(f"\nSample duplicated compound_keys:")
    sample_keys = duplicates['compound_key'].unique()[:5]
    for key in sample_keys:
        count = len(df[df['compound_key'] == key])
        print(f"  {key}: {count} rows")

    if dry_run:
        print("\n[DRY RUN] No changes made.")
        return num_duplicate_rows

    # Remove duplicates
    df_deduped = df.drop_duplicates(subset='compound_key', keep=keep)
    final_count = len(df_deduped)

    print(f"\nAfter deduplication: {final_count:,} rows")
    print(f"Removed: {original_count - final_count:,} rows")

    # Backup original
    backup_path = manifest_path.with_suffix('.backup.parquet')
    import shutil
    shutil.copy(manifest_path, backup_path)
    print(f"Backup saved: {backup_path}")

    # Write deduplicated manifest
    df_deduped.to_parquet(manifest_path, index=False)
    print(f"Updated: {manifest_path}")

    return original_count - final_count


def main():
    parser = argparse.ArgumentParser(
        description="Remove duplicate compound_keys from manifest"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/master/manifest.parquet"),
        help="Path to manifest (default: data/master/manifest.parquet)"
    )
    parser.add_argument(
        "--keep",
        choices=['first', 'last'],
        default='first',
        help="Which duplicate to keep (default: first)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without making changes"
    )

    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    num_removed = deduplicate_manifest(
        manifest_path=args.manifest,
        keep=args.keep,
        dry_run=args.dry_run,
    )

    if num_removed > 0 and not args.dry_run:
        print(f"\nDone. Removed {num_removed:,} duplicate rows.")

    sys.exit(0)


if __name__ == "__main__":
    main()
