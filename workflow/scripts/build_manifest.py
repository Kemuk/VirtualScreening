#!/usr/bin/env python3
"""
build_manifest.py

Merge manifest creation results from array job workers into final manifest.parquet.

This script is called after all manifest array tasks complete. It:
1. Collects all result JSON files from workers
2. Filters successful entries
3. Creates the manifest DataFrame with proper schema
4. Saves to manifest.parquet

Usage:
    python workflow/scripts/build_manifest.py \
        --results-dir data/slurm/results/manifest \
        --output data/master/manifest.parquet
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# 40-column manifest schema (must match create_manifest.py)
MANIFEST_SCHEMA = pa.schema([
    # Identity (5 columns)
    ('ligand_id', pa.string()),
    ('protein_id', pa.string()),
    ('dataset', pa.string()),
    ('compound_key', pa.string()),
    ('is_active', pa.bool_()),

    # Chemistry (2 columns)
    ('smiles_input', pa.string()),
    ('smiles_canonical', pa.string()),

    # Input Sources (3 columns)
    ('source_smiles_file', pa.string()),
    ('source_sdf_path', pa.string()),
    ('source_mol2_path', pa.string()),

    # Preparation (2 columns)
    ('preparation_status', pa.bool_()),
    ('ligand_pdbqt_path', pa.string()),

    # Receptor (2 columns)
    ('receptor_pdbqt_path', pa.string()),
    ('receptor_pdb_path', pa.string()),

    # Docking Box (6 columns)
    ('box_center_x', pa.float64()),
    ('box_center_y', pa.float64()),
    ('box_center_z', pa.float64()),
    ('box_size_x', pa.float64()),
    ('box_size_y', pa.float64()),
    ('box_size_z', pa.float64()),

    # Docking (4 columns)
    ('docking_status', pa.bool_()),
    ('docked_pdbqt_path', pa.string()),
    ('docking_log_path', pa.string()),
    ('vina_score', pa.float64()),

    # Rescoring (14 columns)
    ('rescoring_status', pa.bool_()),
    ('docked_sdf_path', pa.string()),
    ('binding_affinity_pK', pa.float64()),
    ('aev_plig_best_score', pa.float64()),
    ('aev_prediction_0', pa.float64()),
    ('aev_prediction_1', pa.float64()),
    ('aev_prediction_2', pa.float64()),
    ('aev_prediction_3', pa.float64()),
    ('aev_prediction_4', pa.float64()),
    ('aev_prediction_5', pa.float64()),
    ('aev_prediction_6', pa.float64()),
    ('aev_prediction_7', pa.float64()),
    ('aev_prediction_8', pa.float64()),
    ('aev_prediction_9', pa.float64()),

    # Metadata (2 columns)
    ('created_at', pa.timestamp('ns')),
    ('last_updated', pa.timestamp('ns')),
])


def collect_results(results_dir: Path) -> List[Dict]:
    """
    Collect all result JSON files from manifest workers.

    Args:
        results_dir: Directory containing result_*.json files

    Returns:
        List of all result records
    """
    all_results = []

    result_files = sorted(results_dir.glob('result_*.json'))
    print(f"Found {len(result_files)} result files")

    for result_file in result_files:
        with open(result_file) as f:
            chunk_results = json.load(f)
            all_results.extend(chunk_results)

    return all_results


def backup_manifest(manifest_path: Path) -> Path:
    """
    Create a backup of existing manifest.

    Args:
        manifest_path: Path to existing manifest

    Returns:
        Path to backup file
    """
    backup_dir = manifest_path.parent / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"manifest_{timestamp}.parquet"

    # Copy file (don't move, in case something goes wrong)
    import shutil
    shutil.copy2(manifest_path, backup_path)

    return backup_path


def build_manifest(results: List[Dict], output_path: Path) -> int:
    """
    Build manifest.parquet from collected results.

    Args:
        results: List of result dicts from workers
        output_path: Output parquet file path

    Returns:
        Number of entries in manifest
    """
    # Filter successful results
    successful = [r for r in results if r.get('success', True)]
    failed = [r for r in results if not r.get('success', True)]

    if failed:
        print(f"Warning: {len(failed)} items failed processing")
        for f in failed[:5]:  # Show first 5 failures
            print(f"  - {f.get('protein_id', '?')}/{f.get('ligand_id', '?')}: {f.get('error', 'unknown error')}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")

    if not successful:
        raise ValueError("No successful results to build manifest from")

    # Remove 'success' and 'error' keys from results
    entries = []
    for r in successful:
        entry = {k: v for k, v in r.items() if k not in ('success', 'error')}
        entries.append(entry)

    # Convert to DataFrame
    df = pd.DataFrame(entries)

    # Ensure correct data types
    df['is_active'] = df['is_active'].astype(bool)
    df['preparation_status'] = df['preparation_status'].astype(bool)
    df['docking_status'] = df['docking_status'].astype(bool)
    df['rescoring_status'] = df['rescoring_status'].astype(bool)

    # Convert timestamps (they come as ISO strings from JSON)
    # Use ISO8601 format to handle both with and without microseconds
    df['created_at'] = pd.to_datetime(df['created_at'], format='ISO8601')
    df['last_updated'] = pd.to_datetime(df['last_updated'], format='ISO8601')

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to Parquet with compression
    table = pa.Table.from_pandas(df, schema=MANIFEST_SCHEMA)
    pq.write_table(table, output_path, compression='snappy')

    return len(df)


def main():
    parser = argparse.ArgumentParser(
        description="Build manifest.parquet from array job results"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing result_*.json files from manifest workers",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/master/manifest.parquet"),
        help="Output manifest path",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup of existing manifest",
    )

    args = parser.parse_args()

    # Backup existing manifest if present
    if args.output.exists() and not args.no_backup:
        backup_path = backup_manifest(args.output)
        print(f"Backed up existing manifest to: {backup_path}")

    # Collect results
    print(f"Collecting results from: {args.results_dir}")
    results = collect_results(args.results_dir)
    print(f"Total results: {len(results)}")

    if not results:
        print("ERROR: No results found", file=sys.stderr)
        return 1

    # Build manifest
    print(f"Building manifest...")
    n_entries = build_manifest(results, args.output)

    print(f"Manifest created: {args.output}")
    print(f"  Total entries: {n_entries}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
