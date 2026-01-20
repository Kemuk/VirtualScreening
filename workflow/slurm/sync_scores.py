#!/usr/bin/env python3
"""
sync_scores.py

Scan for existing docked files and extract Vina scores into manifest.
Updates docking_status and vina_score without re-docking.

Usage:
    python -m workflow.slurm.sync_scores
    python -m workflow.slurm.sync_scores --dry-run  # preview changes
"""

import argparse
import fcntl
import re
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


# Regex patterns for extracting Vina scores
VINA_LOG_PATTERN = re.compile(r'^\s*1\s+(-?\d+\.?\d*)\s+', re.MULTILINE)
VINA_PDBQT_PATTERN = re.compile(r'REMARK\s+VINA\s+RESULT:\s+(-?\d+\.?\d*)', re.IGNORECASE)


def extract_score_from_log(log_path: Path) -> float:
    """Extract best Vina score from log file."""
    if not log_path.exists():
        return None

    try:
        content = log_path.read_text()
        match = VINA_LOG_PATTERN.search(content)
        if match:
            return float(match.group(1))
    except Exception:
        pass

    return None


def extract_score_from_pdbqt(pdbqt_path: Path) -> float:
    """Extract best Vina score from docked PDBQT file."""
    if not pdbqt_path.exists():
        return None

    try:
        with open(pdbqt_path) as f:
            for line in f:
                if line.startswith('REMARK'):
                    match = VINA_PDBQT_PATTERN.search(line)
                    if match:
                        return float(match.group(1))
                elif line.startswith('ATOM') or line.startswith('HETATM'):
                    break  # Stop after header
    except Exception:
        pass

    return None


def check_item(row: dict) -> dict:
    """
    Check if docked files exist and extract score.

    Args:
        row: Manifest row as dict

    Returns:
        Dict with compound_key, exists, score
    """
    compound_key = row['compound_key']
    docked_path = row.get('docked_pdbqt_path')

    if not docked_path:
        return {'compound_key': compound_key, 'exists': False, 'score': None}

    docked_path = Path(docked_path)

    if not docked_path.exists():
        return {'compound_key': compound_key, 'exists': False, 'score': None}

    # File exists - try to extract score
    score = None

    # First try log file
    log_path = docked_path.parent / "log" / f"{docked_path.stem.replace('_docked', '')}.log"
    score = extract_score_from_log(log_path)

    # Fall back to PDBQT header
    if score is None:
        score = extract_score_from_pdbqt(docked_path)

    return {
        'compound_key': compound_key,
        'exists': True,
        'score': score,
    }


def sync_scores(
    manifest_path: Path,
    dry_run: bool = False,
    max_workers: int = 8,
) -> tuple:
    """
    Sync docking_status and vina_score from existing files.

    Args:
        manifest_path: Path to manifest.parquet
        dry_run: If True, don't write changes
        max_workers: Number of parallel workers

    Returns:
        Tuple of (num_updated_status, num_updated_scores)
    """
    # Load manifest
    df = pq.read_table(manifest_path).to_pandas()
    print(f"Loaded manifest: {len(df):,} rows")

    # Find items that might need syncing
    # Items where docking_status=False but docked file might exist
    needs_check = df[df['docking_status'] == False].copy()
    print(f"Items with docking_status=False: {len(needs_check):,}")

    if needs_check.empty:
        print("Nothing to sync - all items already have docking_status=True")
        return 0, 0

    # Check each item in parallel
    print(f"\nScanning for existing docked files...")
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(check_item, row.to_dict()): row['compound_key']
            for _, row in needs_check.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                compound_key = futures[future]
                results.append({'compound_key': compound_key, 'exists': False, 'score': None})

    # Analyze results
    found_existing = [r for r in results if r['exists']]
    with_scores = [r for r in found_existing if r['score'] is not None]

    print(f"\nResults:")
    print(f"  Found existing docked files: {len(found_existing):,}")
    print(f"  With extractable scores: {len(with_scores):,}")

    if not found_existing:
        print("\nNo existing docked files found that need syncing.")
        return 0, 0

    # Show sample
    print(f"\nSample of found files:")
    for r in found_existing[:5]:
        score_str = f"{r['score']:.2f}" if r['score'] else "N/A"
        print(f"  {r['compound_key']}: score={score_str}")

    if dry_run:
        print(f"\n[DRY RUN] Would update {len(found_existing):,} rows")
        return len(found_existing), len(with_scores)

    # Update manifest
    print(f"\nUpdating manifest...")

    # Create lookup dicts
    status_updates = {r['compound_key'] for r in found_existing}
    score_updates = {r['compound_key']: r['score'] for r in with_scores}

    # Acquire lock and update
    lock_path = manifest_path.with_suffix('.lock')

    with open(lock_path, 'w') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            # Reload to get fresh state
            df = pq.read_table(manifest_path).to_pandas()

            # Update docking_status
            mask = df['compound_key'].isin(status_updates)
            df.loc[mask, 'docking_status'] = True

            # Update vina_score
            for compound_key, score in score_updates.items():
                if score is not None:
                    df.loc[df['compound_key'] == compound_key, 'vina_score'] = score

            # Atomic write
            temp_path = manifest_path.with_suffix('.tmp')
            df.to_parquet(temp_path, index=False)
            temp_path.rename(manifest_path)

            print(f"Updated manifest: {manifest_path}")

        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return len(found_existing), len(with_scores)


def main():
    parser = argparse.ArgumentParser(
        description="Sync docking_status and vina_score from existing files"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/master/manifest.parquet"),
        help="Path to manifest (default: data/master/manifest.parquet)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )

    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    num_status, num_scores = sync_scores(
        manifest_path=args.manifest,
        dry_run=args.dry_run,
        max_workers=args.workers,
    )

    print(f"\nDone. Updated {num_status:,} status flags, {num_scores:,} scores.")
    sys.exit(0)


if __name__ == "__main__":
    main()
