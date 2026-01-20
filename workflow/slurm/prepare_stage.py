#!/usr/bin/env python3
"""
prepare_stage.py

Filter manifest to pending items for a stage and write subset parquet.

Usage:
    python -m workflow.slurm.prepare_stage --stage docking --num-chunks 500
    python -m workflow.slurm.prepare_stage --stage docking --max-items 1000 --num-chunks 5  # devel

Output:
    - data/master/pending/docking.parquet (filtered subset)
    - Prints suggested sbatch command
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from workflow.slurm.stage_config import get_stage_config, list_stages


def filter_pending(
    manifest_path: Path,
    stage: str,
) -> pd.DataFrame:
    """
    Filter manifest to items pending processing for a stage.

    Args:
        manifest_path: Path to manifest.parquet
        stage: Stage name

    Returns:
        DataFrame with pending items
    """
    config = get_stage_config(stage)

    # Load manifest
    df = pq.read_table(manifest_path).to_pandas()
    original_count = len(df)

    # Filter by dependency (previous stage must be complete)
    depends_on = config.get('depends_on')
    if depends_on:
        df = df[df[depends_on] == True]

    # Filter by status column (this stage must be incomplete)
    status_col = config.get('status_column')
    if status_col:
        df = df[df[status_col] == False]

    # Check file existence requirement (e.g., conversion needs SDF to not exist,
    # aev_infer needs SDF to exist)
    check_file = config.get('check_file_column')
    if check_file:
        def file_exists(path_str):
            if pd.isna(path_str) or not path_str:
                return False
            return Path(path_str).exists()

        if status_col:
            # Stage has status column: require file to EXIST (e.g., aev_infer needs SDF)
            df = df[df[check_file].apply(file_exists)]
        else:
            # Stage without status column: require file to NOT exist (e.g., conversion)
            df = df[~df[check_file].apply(file_exists)]

    print(f"Stage: {stage} ({config['description']})")
    print(f"  Total in manifest: {original_count:,}")
    print(f"  Pending: {len(df):,}")

    return df


def prepare_stage(
    manifest_path: Path,
    stage: str,
    num_chunks: int,
    output_dir: Path,
    max_items: int = None,
) -> Path:
    """
    Prepare stage by filtering manifest and writing pending subset.

    Args:
        manifest_path: Path to manifest.parquet
        stage: Stage name
        num_chunks: Number of chunks for array job
        output_dir: Directory for pending parquet files
        max_items: Maximum items to process (for devel testing)

    Returns:
        Path to pending parquet file
    """
    # Filter to pending items
    pending_df = filter_pending(manifest_path, stage)

    if len(pending_df) == 0:
        print(f"\nNothing to do - all items complete for stage '{stage}'")
        return None

    # Limit items for devel mode
    if max_items is not None and len(pending_df) > max_items:
        pending_df = pending_df.head(max_items)
        print(f"  Limited to: {max_items:,} items (--max-items)")

    # Create output directory
    pending_dir = output_dir / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    # Write pending parquet using pyarrow directly for compatibility
    # This avoids "Repetition level histogram size mismatch" errors
    output_path = pending_dir / f"{stage}.parquet"
    table = pa.Table.from_pandas(pending_df, preserve_index=False)
    pq.write_table(
        table,
        output_path,
        use_dictionary=False,  # Avoid dict encoding issues
        write_statistics=False,  # Avoid histogram issues
    )

    print(f"\nWrote: {output_path}")
    print(f"  Rows: {len(pending_df):,}")

    # Calculate chunk info
    total_rows = len(pending_df)
    actual_chunks = min(num_chunks, total_rows)
    items_per_chunk = (total_rows + actual_chunks - 1) // actual_chunks

    print(f"\nChunk info:")
    print(f"  Requested chunks: {num_chunks}")
    print(f"  Actual chunks: {actual_chunks} (capped to row count)")
    print(f"  Items per chunk: ~{items_per_chunk}")

    # Print suggested sbatch command
    print(f"\nSubmit with:")
    print(f"  sbatch --array=0-{actual_chunks - 1} workflow/slurm/{stage}.slurm")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare stage by filtering manifest to pending items"
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=list_stages(),
        help="Stage to prepare"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/master/manifest.parquet"),
        help="Path to manifest (default: data/master/manifest.parquet)"
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=500,
        help="Number of chunks for array job (default: 500)"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum items to process (for devel testing)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/master"),
        help="Output directory (default: data/master)"
    )

    args = parser.parse_args()

    # Validate manifest exists
    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    # Prepare stage
    output_path = prepare_stage(
        manifest_path=args.manifest,
        stage=args.stage,
        num_chunks=args.num_chunks,
        output_dir=args.output_dir,
        max_items=args.max_items,
    )

    if output_path is None:
        sys.exit(0)  # Nothing to do is not an error

    sys.exit(0)


if __name__ == "__main__":
    main()
