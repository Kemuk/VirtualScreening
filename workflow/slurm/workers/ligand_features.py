"""
ligand_features.py

Stage 1 worker: per-ligand feature generation.

Reads a slice of the pending parquet, generates feature vectors for each ligand
using the configured method, and writes a parquet with feature data and status.

Usage (SLURM array task):
    python -m workflow.slurm.workers.ligand_features \\
        --pending      data/master/pending/ligand_features.parquet \\
        --task-id      ${SLURM_ARRAY_TASK_ID} \\
        --num-chunks   ${NUM_CHUNKS} \\
        --config       config/config.yaml \\
        --results-dir  data/master/results/ligand_features \\
        --cache-dir    data/ligand_based/cache

Usage (merge after all array tasks complete):
    python -m workflow.slurm.workers.ligand_features --merge \\
        --results-dir  data/master/results/ligand_features \\
        --cache-dir    data/ligand_based/cache
"""

import argparse
import sys
from pathlib import Path

import polars as pl
import yaml

from workflow.slurm.workers import read_slice_pl
from workflow.ligand_based.registry import get_method


def process_slice(
    pending_path: Path,
    task_id: int,
    num_chunks: int,
    config_path: Path,
    results_dir: Path,
    cache_dir: Path,
) -> int:
    cfg = yaml.safe_load(open(config_path)).get("ligand_based", {})
    method_name = cfg.get("method", "usrcat")
    method = get_method(method_name)

    df = read_slice_pl(pending_path, task_id, num_chunks)
    if df.is_empty():
        return 0

    feature_vecs = method.generate_features_batch(df, cache_dir, cfg)

    out = (
        df.with_columns(feature_vecs.alias("feature_vec"))
        .with_columns(pl.lit(method_name).alias("method"))
        .with_columns(
            pl.col("feature_vec").is_not_null().alias("success"),
            pl.when(pl.col("feature_vec").is_null())
            .then(pl.lit("feature_generation_failed"))
            .otherwise(pl.lit(None))
            .alias("error"),
        )
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    out.write_parquet(results_dir / f"ligand_features_{task_id:05d}.parquet")
    n_ok = out["success"].sum()
    print(f"Task {task_id}: {n_ok}/{len(out)} features generated")
    return len(out)


def merge(results_dir: Path, cache_dir: Path) -> None:
    """Concatenate per-task parquets into a single features.parquet cache."""
    chunks = sorted(results_dir.glob("ligand_features_?????.parquet"))
    if not chunks:
        print("No result files found — nothing to merge.", file=sys.stderr)
        sys.exit(1)

    print(f"Merging {len(chunks)} result files...")
    features = (
        pl.concat([pl.read_parquet(f) for f in chunks])
        .filter(pl.col("success"))
        .drop(["success", "error"])
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "features.parquet"
    features.write_parquet(out_path)
    print(f"Written: {out_path}  ({len(features):,} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Ligand features worker — Stage 1 of ligand-based screening",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--merge", action="store_true",
                        help="Run merge mode: concat per-task parquets into features.parquet")
    parser.add_argument("--pending", type=Path,
                        help="Path to pending parquet (worker mode)")
    parser.add_argument("--task-id", type=int,
                        help="SLURM_ARRAY_TASK_ID (worker mode)")
    parser.add_argument("--num-chunks", type=int,
                        help="Total number of array tasks (worker mode)")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"),
                        help="Path to config.yaml")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory for per-task result parquets")
    parser.add_argument("--cache-dir", type=Path, required=True,
                        help="Directory for the merged features.parquet cache")

    args = parser.parse_args()

    if args.merge:
        merge(args.results_dir, args.cache_dir)
        return

    for flag in ("pending", "task_id", "num_chunks", "config"):
        if getattr(args, flag) is None:
            parser.error(f"--{flag.replace('_', '-')} is required in worker mode")

    process_slice(
        pending_path=args.pending,
        task_id=args.task_id,
        num_chunks=args.num_chunks,
        config_path=args.config,
        results_dir=args.results_dir,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
