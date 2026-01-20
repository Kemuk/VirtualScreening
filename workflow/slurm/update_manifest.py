#!/usr/bin/env python3
"""
update_manifest.py

Merge results from worker CSV files back into the manifest.

Usage:
    python -m workflow.slurm.update_manifest --stage docking

Input:
    - data/master/manifest.parquet
    - data/master/results/{stage}_*.csv

Output:
    - data/master/manifest.parquet (updated)
"""

import argparse
import fcntl
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from workflow.slurm.stage_config import get_stage_config, list_stages


def load_results(results_dir: Path, stage: str) -> pd.DataFrame:
    """
    Load and concatenate all result CSV files for a stage.

    Args:
        results_dir: Directory containing result files
        stage: Stage name

    Returns:
        DataFrame with all results
    """
    pattern = f"{stage}_*.csv"
    result_files = sorted(results_dir.glob(pattern))

    if not result_files:
        print(f"No result files found matching: {results_dir / pattern}")
        return pd.DataFrame()

    print(f"Found {len(result_files)} result files")

    # Load and concatenate
    dfs = []
    for f in result_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"WARNING: Failed to read {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    results = pd.concat(dfs, ignore_index=True)
    print(f"  Total results: {len(results):,}")

    return results


def update_manifest(
    manifest_path: Path,
    stage: str,
    results_dir: Path,
) -> int:
    """
    Update manifest with results from worker CSV files.

    Uses file locking to prevent concurrent writes.

    Args:
        manifest_path: Path to manifest.parquet
        stage: Stage name
        results_dir: Directory containing result CSV files

    Returns:
        Number of items updated
    """
    config = get_stage_config(stage)
    status_col = config.get('status_column')
    score_col = config.get('score_column')

    # Load results
    results = load_results(results_dir, stage)

    if results.empty:
        print("No results to merge")
        return 0

    # Filter to successful results
    if 'success' in results.columns:
        successful = results[results['success'] == True]
        failed = results[results['success'] == False]
        print(f"  Successful: {len(successful):,}")
        print(f"  Failed: {len(failed):,}")
    else:
        successful = results
        failed = pd.DataFrame()

    if successful.empty:
        print("No successful results to merge")
        return 0

    # Acquire lock and update manifest
    lock_path = manifest_path.with_suffix('.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, 'w') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            # Load manifest
            manifest = pq.read_table(manifest_path).to_pandas()
            print(f"\nManifest loaded: {len(manifest):,} rows")

            # Get compound_keys that succeeded
            completed_keys = set(successful['compound_key'])

            # Update status column
            if status_col:
                mask = manifest['compound_key'].isin(completed_keys)
                manifest.loc[mask, status_col] = True
                print(f"  Updated {status_col}: {mask.sum():,} rows")

            # Update score column if present in results
            if score_col and 'score' in successful.columns:
                # Create mapping from compound_key to score
                score_map = successful.set_index('compound_key')['score'].to_dict()

                # Update scores
                def get_score(key):
                    return score_map.get(key)

                mask = manifest['compound_key'].isin(completed_keys)
                manifest.loc[mask, score_col] = manifest.loc[mask, 'compound_key'].apply(get_score)
                print(f"  Updated {score_col}: {mask.sum():,} rows")

            # Atomic write
            temp_path = manifest_path.with_suffix('.tmp')
            manifest.to_parquet(temp_path, index=False)
            temp_path.rename(manifest_path)

            print(f"\nManifest updated: {manifest_path}")
            return len(completed_keys)

        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(
        description="Merge worker results back into manifest"
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=list_stages(),
        help="Stage to update"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/master/manifest.parquet"),
        help="Path to manifest (default: data/master/manifest.parquet)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/master/results"),
        help="Directory containing result CSV files (default: data/master/results)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    if not args.results_dir.exists():
        print(f"ERROR: Results directory not found: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    # Update manifest
    num_updated = update_manifest(
        manifest_path=args.manifest,
        stage=args.stage,
        results_dir=args.results_dir,
    )

    print(f"\nDone. Updated {num_updated:,} items.")
    sys.exit(0)


if __name__ == "__main__":
    main()
