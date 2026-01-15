#!/usr/bin/env python3
"""
shard_csv.py

Split a CSV file into N shards for parallel processing.
Each shard contains rows where row_index % num_shards == shard_id.

Usage:
    python shard_csv.py --input data.csv --num-shards 100 --outdir shards/

Output:
    shards/data_shard_0.csv
    shards/data_shard_1.csv
    ...
    shards/data_shard_99.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd


def shard_csv(
    input_path: Path,
    num_shards: int,
    outdir: Path,
    prefix: str = None,
) -> list:
    """
    Split CSV into N shards.

    Args:
        input_path: Path to input CSV
        num_shards: Number of shards to create
        outdir: Output directory for shards
        prefix: Prefix for output files (default: input filename without extension)

    Returns:
        List of output file paths
    """
    # Read input CSV
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    total_rows = len(df)
    print(f"  Total rows: {total_rows}")

    if total_rows == 0:
        print("ERROR: Input CSV is empty", file=sys.stderr)
        return []

    # Determine prefix
    if prefix is None:
        prefix = input_path.stem

    # Create output directory
    outdir.mkdir(parents=True, exist_ok=True)

    # Create shards
    output_paths = []
    rows_per_shard = []

    for shard_id in range(num_shards):
        # Select rows for this shard
        shard_df = df.iloc[shard_id::num_shards]

        if len(shard_df) == 0:
            # Skip empty shards (can happen if num_shards > total_rows)
            continue

        # Write shard
        output_path = outdir / f"{prefix}_shard_{shard_id}.csv"
        shard_df.to_csv(output_path, index=False)

        output_paths.append(output_path)
        rows_per_shard.append(len(shard_df))

    print(f"\nCreated {len(output_paths)} shards in {outdir}")
    print(f"  Rows per shard: {min(rows_per_shard)} - {max(rows_per_shard)}")
    print(f"  Total rows across shards: {sum(rows_per_shard)}")

    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description="Split CSV into shards for parallel processing"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV file"
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        required=True,
        help="Number of shards to create"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for shards"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix for output files (default: input filename)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.num_shards < 1:
        print(f"ERROR: num-shards must be >= 1", file=sys.stderr)
        sys.exit(1)

    # Shard the CSV
    output_paths = shard_csv(
        input_path=args.input,
        num_shards=args.num_shards,
        outdir=args.outdir,
        prefix=args.prefix,
    )

    if not output_paths:
        print("ERROR: No shards created", file=sys.stderr)
        sys.exit(1)

    print(f"\nSharding complete!")
    sys.exit(0)


if __name__ == "__main__":
    main()
