#!/usr/bin/env python3
"""
merge_stage_results.py

Merge stage chunk results into the manifest using compound_key.
"""

import argparse
from datetime import datetime
from pathlib import Path

import polars as pl


STAGE_COLUMNS = {
    "preparation": ["preparation_status"],
    "docking": ["docking_status", "vina_score"],
    "conversion": ["conversion_status"],
}


def read_results(results_dir: Path) -> pl.DataFrame:
    files = sorted(results_dir.glob("chunk_*.csv"))
    if not files:
        return pl.DataFrame()
    frames = [pl.read_csv(path) for path in files]
    return pl.concat(frames, how="vertical")


def merge_results(manifest: pl.DataFrame, results: pl.DataFrame, stage: str) -> pl.DataFrame:
    if results.is_empty():
        return manifest

    if "compound_key" not in results.columns:
        raise ValueError("Results must include compound_key")

    if "conversion_status" in results.columns and "conversion_status" not in manifest.columns:
        manifest = manifest.with_columns(pl.lit(False).alias("conversion_status"))

    updates = [col for col in STAGE_COLUMNS[stage] if col in results.columns]
    join_df = manifest.join(results, on="compound_key", how="left", suffix="_result")

    update_exprs = []
    update_flags = []
    for col in updates:
        result_col = f"{col}_result"
        update_exprs.append(
            pl.when(pl.col(result_col).is_not_null())
            .then(pl.col(result_col))
            .otherwise(pl.col(col))
            .alias(col)
        )
        update_flags.append(pl.col(result_col).is_not_null())

    if update_exprs:
        joined = join_df.with_columns(update_exprs)
        update_mask = update_flags[0]
        for flag in update_flags[1:]:
            update_mask = update_mask | flag
        joined = joined.with_columns(
            pl.when(update_mask)
            .then(pl.lit(datetime.now()))
            .otherwise(pl.col("last_updated"))
            .alias("last_updated")
        )
    else:
        joined = join_df

    drop_cols = [c for c in joined.columns if c.endswith("_result") or c == "error"]
    return joined.drop(drop_cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge stage results into manifest")
    parser.add_argument("--stage", choices=sorted(STAGE_COLUMNS.keys()), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)

    args = parser.parse_args()

    manifest = pl.read_parquet(args.manifest)
    results = read_results(args.results_dir)
    updated = merge_results(manifest, results, args.stage)

    updated.write_parquet(args.manifest)
    print(f"Updated manifest with {len(results)} {args.stage} results")


if __name__ == "__main__":
    main()
