#!/usr/bin/env python3
"""
shard_stage.py

Create per-stage chunk CSVs from the manifest for array-style execution.
"""

import argparse
from pathlib import Path

import polars as pl


STAGE_COLUMNS = {
    "preparation": [
        "compound_key",
        "ligand_id",
        "protein_id",
        "smiles_input",
        "ligand_pdbqt_path",
    ],
    "docking": [
        "compound_key",
        "ligand_id",
        "protein_id",
        "ligand_pdbqt_path",
        "receptor_pdbqt_path",
        "docked_pdbqt_path",
    ],
    "conversion": [
        "compound_key",
        "ligand_id",
        "docked_pdbqt_path",
        "docked_sdf_path",
    ],
}


def load_manifest(manifest_path: Path) -> pl.DataFrame:
    """Load manifest into a Polars DataFrame."""
    return pl.read_parquet(manifest_path)


def add_conversion_status(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure conversion_status exists by checking docked_sdf_path."""
    if "conversion_status" in df.columns:
        return df
    paths = df.get_column("docked_sdf_path").to_list()
    statuses = [Path(path).exists() if path else False for path in paths]
    return df.with_columns(pl.Series("conversion_status", statuses))


def filter_stage(df: pl.DataFrame, stage: str) -> pl.DataFrame:
    """Filter manifest rows based on stage status."""
    if stage == "preparation":
        return df.filter(pl.col("preparation_status") == False)
    if stage == "docking":
        return df.filter(
            (pl.col("preparation_status") == True)
            & (pl.col("docking_status") == False)
        )
    if stage == "conversion":
        df = add_conversion_status(df)
        return df.filter(
            (pl.col("docking_status") == True)
            & (pl.col("conversion_status") == False)
        )
    raise ValueError(f"Unknown stage: {stage}")


def shard_stage(
    manifest_path: Path,
    stage: str,
    outdir: Path,
    num_chunks: int,
    max_items: int | None = None,
) -> list[Path]:
    """Create chunk CSV files for a stage."""
    df = load_manifest(manifest_path)
    df = filter_stage(df, stage)

    if max_items:
        df = df.head(max_items)

    columns = STAGE_COLUMNS[stage]
    df = df.select([col for col in columns if col in df.columns])

    df = df.with_row_count("row_idx")
    df = df.with_columns((pl.col("row_idx") % num_chunks).alias("chunk_id"))

    outdir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for chunk_id in range(num_chunks):
        chunk_df = df.filter(pl.col("chunk_id") == chunk_id).drop(["row_idx", "chunk_id"])
        output_path = outdir / f"chunk_{chunk_id}.csv"
        chunk_df.write_csv(output_path)
        output_paths.append(output_path)

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Shard manifest rows for a stage")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGE_COLUMNS.keys()), required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--max-items", type=int)

    args = parser.parse_args()

    if args.num_chunks < 1:
        raise ValueError("num-chunks must be >= 1")

    paths = shard_stage(
        manifest_path=args.manifest,
        stage=args.stage,
        outdir=args.outdir,
        num_chunks=args.num_chunks,
        max_items=args.max_items,
    )
    print(f"Created {len(paths)} chunks in {args.outdir}")


if __name__ == "__main__":
    main()
