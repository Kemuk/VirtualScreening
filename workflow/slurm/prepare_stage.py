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

import polars as pl
from tqdm import tqdm

from workflow.slurm.stage_config import get_stage_config, list_stages


def filter_pending(
    manifest_path: Path,
    stage: str,
) -> pl.DataFrame:
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
    df = pl.read_parquet(manifest_path)
    original_count = df.height

    # Filter by dependency (previous stage must be complete)
    depends_on = config.get('depends_on')
    if depends_on:
        df = df.filter(pl.col(depends_on) == True)

    status_col = config.get('status_column')
    check_file = config.get('check_file_column')

    def file_missing(path_str):
        if path_str is None or path_str == "":
            return True
        return not Path(path_str).exists()

    if status_col and check_file:
        paths = df.select(check_file).to_series().to_list()
        missing_mask = [
            file_missing(p)
            for p in tqdm(paths, desc="Checking files", unit="file")
        ]

        df = df.with_columns(
            pl.Series("_file_missing", missing_mask)
        )

        df = df.filter(
            (pl.col(status_col) == False) | pl.col("_file_missing")
        ).drop("_file_missing")

    elif status_col:
        df = df.filter(pl.col(status_col) == False)

    elif check_file:
        paths = df.select(check_file).to_series().to_list()
        missing_mask = [
            file_missing(p)
            for p in tqdm(paths, desc="Checking files", unit="file")
        ]

        df = df.with_columns(
            pl.Series("_file_missing", missing_mask)
        )

        df = df.filter(pl.col("_file_missing")).drop("_file_missing")

    print(f"Stage: {stage} ({config['description']})")
    print(f"  Total in manifest: {original_count:,}")
    print(f"  Pending: {df.height:,}")

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

    if pending_df.height == 0:
        print(f"\nNothing to do - all items complete for stage '{stage}'")
        return None

    # Limit items for devel mode
    if max_items is not None and pending_df.height > max_items:
        pending_df = pending_df.head(max_items)
        print(f"  Limited to: {max_items:,} items (--max-items)")

    # Create output directory
    pending_dir = output_dir / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    # Write pending parquet
    output_path = pending_dir / f"{stage}.parquet"
    pending_df.write_parquet(output_path)

    print(f"\nWrote: {output_path}")
    print(f"  Rows: {pending_df.height:,}")

    # Calculate chunk info
    total_rows = pending_df.height
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