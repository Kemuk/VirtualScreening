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
    config = get_stage_config(stage)

    # Lazy scan
    lf = pl.scan_parquet(manifest_path)

    original_count = lf.select(pl.count()).collect().item()

    # Dependency filter (guard: column may not exist yet for new stages)
    depends_on = config.get("depends_on")
    schema = lf.schema
    if depends_on:
        if depends_on in schema:
            lf = lf.filter(pl.col(depends_on))
        # else: dependency column absent → treat all rows as eligible

    status_col = config.get("status_column")
    check_file = config.get("check_file_column")

    # Build file-missing expression (vectorised via Polars)
    file_missing_expr = None
    if check_file:
        file_missing_expr = (
            pl.when(pl.col(check_file).is_null() | (pl.col(check_file) == ""))
            .then(True)
            .otherwise(
                pl.col(check_file).map_elements(
                    lambda p: not Path(p).exists(),
                    return_dtype=pl.Boolean,
                )
            )
        )

    # Build pending condition (guard: status column may not exist yet)
    if status_col and check_file:
        if status_col in schema:
            pending_expr = (~pl.col(status_col)) | file_missing_expr
        else:
            pending_expr = file_missing_expr
    elif status_col:
        pending_expr = ~pl.col(status_col) if status_col in schema else pl.lit(True)
    elif check_file:
        pending_expr = file_missing_expr
    else:
        pending_expr = pl.lit(True)

    df = lf.filter(pending_expr).collect()

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