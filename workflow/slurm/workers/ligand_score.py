"""
ligand_score.py

Stage 2 worker: per-ligand template similarity scoring.

Reads a slice of the pending parquet (which must already have 'template_smiles'
and 'template_source' columns, added by --prepare), loads features.parquet once,
joins feature vectors for both ligand and template, then computes row-wise
similarity via the configured method.

Usage (prepare — add template columns to pending parquet before the array job):
    python -m workflow.slurm.workers.ligand_score --prepare \\
        --pending      data/master/pending/ligand_score.parquet \\
        --manifest     data/master/manifest.parquet \\
        --config       config/config.yaml

Usage (SLURM array task):
    python -m workflow.slurm.workers.ligand_score \\
        --pending      data/master/pending/ligand_score.parquet \\
        --task-id      ${SLURM_ARRAY_TASK_ID} \\
        --num-chunks   ${NUM_CHUNKS} \\
        --config       config/config.yaml \\
        --results-dir  data/master/results/ligand_score \\
        --cache-dir    data/ligand_based/cache

Usage (merge after all array tasks complete):
    python -m workflow.slurm.workers.ligand_score --merge \\
        --results-dir  data/master/results/ligand_score \\
        --manifest     data/master/manifest.parquet \\
        --output       data/master/manifest_ligand_based.parquet
"""

import argparse
import sys
from pathlib import Path

import polars as pl
import yaml

from workflow.slurm.workers import read_slice_pl
from workflow.ligand_based.registry import get_method


def add_templates(pending_path: Path, manifest_path: Path, explicit_templates: dict) -> None:
    """Add template_smiles and template_source columns to the pending parquet."""
    manifest = pl.read_parquet(
        manifest_path,
        columns=["compound_key", "protein_id", "smiles_canonical", "is_active"],
    )

    # Sort: actives first, then by compound_key — group_by().first() picks the best template
    template_df = (
        manifest
        .sort(["is_active", "compound_key"], descending=[True, False])
        .group_by("protein_id")
        .first()
        .select(["protein_id", "smiles_canonical", "is_active"])
        .rename({"smiles_canonical": "template_smiles"})
        .with_columns(
            pl.when(pl.col("is_active"))
            .then(pl.lit("first_active"))
            .otherwise(pl.lit("first_ligand"))
            .alias("template_source")
        )
        .drop("is_active")
    )

    if explicit_templates:
        config_df = pl.DataFrame({
            "protein_id":      list(explicit_templates.keys()),
            "template_smiles": list(explicit_templates.values()),
            "template_source": ["config"] * len(explicit_templates),
        })
        template_df = template_df.update(config_df, on="protein_id")

    (
        pl.read_parquet(pending_path)
        .join(template_df, on="protein_id", how="left")
        .write_parquet(pending_path)
    )
    print(f"Templates added to {pending_path}")


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

    # Load Stage-1 feature files directly — no pre-merged features.parquet required
    feature_files = sorted(results_dir.glob("ligand_features_?????.parquet"))
    if not feature_files:
        raise FileNotFoundError(
            f"No ligand_features result files found in {results_dir}. "
            "Run Stage 1 (ligand_features) before Stage 2 (ligand_score)."
        )
    features = (
        pl.concat([pl.read_parquet(f) for f in feature_files])
        .filter(pl.col("success").cast(pl.Boolean))
    )

    # Derive template per protein_id unless --prepare already added template_smiles
    if "template_smiles" not in df.columns:
        explicit = cfg.get("templates", {}) or {}
        templates = (
            features
            .filter(pl.col("feature_vec").is_not_null())
            .sort(["is_active", "compound_key"], descending=[True, False])
            .group_by("protein_id")
            .first()
            .select(["protein_id", "smiles_canonical", "is_active"])
            .rename({"smiles_canonical": "template_smiles"})
            .with_columns(
                pl.when(pl.col("is_active"))
                .then(pl.lit("first_active"))
                .otherwise(pl.lit("first_ligand"))
                .alias("template_source")
            )
            .drop("is_active")
        )
        if explicit:
            override_df = pl.DataFrame({
                "protein_id":      list(explicit.keys()),
                "template_smiles": list(explicit.values()),
                "template_source": ["config"] * len(explicit),
            })
            templates = templates.update(override_df, on="protein_id")
        df = df.join(templates, on="protein_id", how="left")

    # Join ligand feature vectors
    df = df.join(features.select(["compound_key", "feature_vec"]), on="compound_key", how="left")

    # Join template feature vectors via smiles_canonical → template_vec
    tmpl_features = (
        features.drop_nulls("feature_vec")
        .select(["smiles_canonical", "feature_vec"])
        .rename({"smiles_canonical": "template_smiles", "feature_vec": "template_vec"})
    )
    df = df.join(tmpl_features, on="template_smiles", how="left")

    # Split valid / invalid rows, score valid, reunite
    valid_mask = pl.col("feature_vec").is_not_null() & pl.col("template_vec").is_not_null()
    valid_df   = df.filter(valid_mask)
    invalid_df = df.filter(~valid_mask)

    if len(valid_df) > 0:
        # pl.Array columns → 2D numpy directly (no custom helper needed)
        sim_vals = method.compute_similarity(
            valid_df["feature_vec"].to_numpy(),
            valid_df["template_vec"].to_numpy(),
            cfg,
        )
        valid_df = valid_df.with_columns(
            pl.Series("ligand_based_score", sim_vals, dtype=pl.Float32)
        )

    invalid_df = invalid_df.with_columns(
        pl.lit(None, dtype=pl.Float32).alias("ligand_based_score")
    )
    df = pl.concat([valid_df, invalid_df])

    out = (
        df.drop(["feature_vec", "template_vec"])
        .with_columns(
            pl.lit(method_name).alias("ligand_based_method"),
            pl.col("ligand_based_score").is_not_null().alias("success"),
            pl.when(pl.col("ligand_based_score").is_null())
            .then(pl.lit("no_features"))
            .otherwise(pl.lit(None))
            .alias("error"),
        )
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    out.write_parquet(results_dir / f"ligand_score_{task_id:05d}.parquet")
    n_ok = out["success"].sum()
    print(f"Task {task_id}: {n_ok}/{len(out)} ligands scored")
    return len(out)


def merge(results_dir: Path, manifest_path: Path, output_path: Path) -> None:
    """Concatenate score parquets and join with manifest."""
    chunks = sorted(results_dir.glob("ligand_score_?????.parquet"))
    if not chunks:
        print("No result files found — nothing to merge.", file=sys.stderr)
        sys.exit(1)

    print(f"Merging {len(chunks)} result files...")
    scores = pl.concat([pl.read_parquet(f) for f in chunks])

    score_cols = ["compound_key", "ligand_based_score", "template_smiles",
                  "template_source", "ligand_based_method"]

    result = (
        pl.read_parquet(manifest_path)
        .join(scores.select(score_cols), on="compound_key", how="left")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path)
    n_scored = scores["success"].sum() if "success" in scores.columns else len(scores)
    print(f"Written: {output_path}  ({n_scored:,} ligands scored)")


def main():
    parser = argparse.ArgumentParser(
        description="Ligand score worker — Stage 2 of ligand-based screening",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true",
                      help="Add template_smiles/template_source columns to pending parquet")
    mode.add_argument("--merge", action="store_true",
                      help="Merge per-task parquets into final manifest_ligand_based.parquet")

    parser.add_argument("--pending", type=Path,
                        help="Path to pending parquet")
    parser.add_argument("--task-id", type=int, default=0,
                        help="SLURM_ARRAY_TASK_ID (worker mode, default: 0)")
    parser.add_argument("--num-chunks", type=int, default=1,
                        help="Total number of array tasks (worker mode, default: 1 = process all)")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"),
                        help="Path to config.yaml")
    parser.add_argument("--results-dir", type=Path,
                        help="Directory for per-task result parquets")
    parser.add_argument("--cache-dir", type=Path,
                        help="Directory for optional cache files (accepted but not required)")
    parser.add_argument("--manifest", type=Path,
                        help="Path to manifest.parquet (prepare / merge mode)")
    parser.add_argument("--output", type=Path,
                        help="Output path for manifest_ligand_based.parquet (merge mode)")
    parser.add_argument("--templates", type=str, default="{}",
                        help="JSON dict of {protein_id: template_smiles} overrides (prepare mode)")

    args = parser.parse_args()

    if args.prepare:
        import json
        for flag in ("pending", "manifest"):
            if getattr(args, flag) is None:
                parser.error(f"--{flag} is required in --prepare mode")
        explicit = json.loads(args.templates)
        add_templates(args.pending, args.manifest, explicit)
        return

    if args.merge:
        for flag in ("results_dir", "manifest", "output"):
            if getattr(args, flag) is None:
                parser.error(f"--{flag.replace('_', '-')} is required in --merge mode")
        merge(args.results_dir, args.manifest, args.output)
        return

    # Worker mode
    for flag in ("pending", "results_dir"):
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
