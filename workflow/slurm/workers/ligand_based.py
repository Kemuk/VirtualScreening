"""
ligand_based.py

Standalone CLI for ligand-based virtual screening (no SLURM required).

Orchestrates both stages in sequence:
  Stage 1 — generate per-ligand feature vectors
  Stage 2 — score each ligand against its target's template molecule

Writes the final merged parquet with ligand_based_score, template_smiles,
template_source, and ligand_based_method columns appended to the manifest.

Usage
-----
All targets:
    python -m workflow.slurm.workers.ligand_based \\
        --manifest  data/master/manifest.parquet \\
        --output    data/master/manifest_ligand_based.parquet \\
        --config    config/config.yaml

Single target (devel / quick test):
    python -m workflow.slurm.workers.ligand_based \\
        --manifest  data/master/manifest.parquet \\
        --output    /tmp/manifest_lb_ADRB2.parquet \\
        --target    ADRB2 \\
        --config    config/config.yaml

First 100 ligands per target:
    python -m workflow.slurm.workers.ligand_based \\
        --manifest       data/master/manifest.parquet \\
        --output         /tmp/manifest_lb_test100.parquet \\
        --max-per-target 100 \\
        --config         config/config.yaml
"""

import argparse
from pathlib import Path

import polars as pl
import yaml

from workflow.ligand_based.registry import get_method


def run(
    manifest_path: Path,
    output_path: Path,
    work_dir: Path,
    method_name: str,
    cfg: dict,
    targets: list = None,
    max_per_target: int = None,
):
    """
    Score all ligands in the manifest and write to output_path.

    Args:
        manifest_path:  Path to input manifest.parquet
        output_path:    Path to write manifest_ligand_based.parquet
        work_dir:       Root for per-target intermediates (conformers, vectors)
        method_name:    Registry key for the LigandBasedMethod to use
        cfg:            ligand_based config dict (from config.yaml)
        targets:        Optional list of protein_id values to restrict processing
        max_per_target: If set, keep only the first N rows per target (for testing)
    """
    method = get_method(method_name)

    df = pl.read_parquet(manifest_path)

    if targets:
        df = df.filter(pl.col("protein_id").is_in(targets))

    if max_per_target is not None:
        df = df.with_row_index("__row").group_by("protein_id").head(max_per_target).sort("__row").drop("__row")

    # ── Stage 1: generate feature vectors ──────────────────────────────────
    print(f"Stage 1: generating {method_name} features for {len(df):,} ligands...")
    feature_vecs = method.generate_features_batch(df, work_dir, cfg)
    df = df.with_columns(feature_vecs.alias("feature_vec"))

    n_ok = df["feature_vec"].is_not_null().sum()
    print(f"  {n_ok:,}/{len(df):,} features generated")

    # ── Determine templates (first active; fall back to first ligand) ───────
    template_df = (
        df.filter(pl.col("feature_vec").is_not_null())
        .sort(["is_active", "ligand_id"], descending=[True, False])
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
    df = df.join(template_df, on="protein_id", how="left")

    # ── Stage 2: score ligands against templates ────────────────────────────
    print("Stage 2: scoring ligands...")
    tmpl_features = (
        df.filter(pl.col("feature_vec").is_not_null())
        .select(["smiles_canonical", "feature_vec"])
        .unique("smiles_canonical")
        .rename({"smiles_canonical": "template_smiles", "feature_vec": "template_vec"})
    )
    df = df.join(tmpl_features, on="template_smiles", how="left")

    valid_mask = pl.col("feature_vec").is_not_null() & pl.col("template_vec").is_not_null()
    valid_df   = df.filter(valid_mask)
    invalid_df = df.filter(~valid_mask)

    if len(valid_df) > 0:
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

    result = (
        df.drop(["feature_vec", "template_vec"])
        .with_columns(pl.lit(method_name).alias("ligand_based_method"))
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path)
    n_scored = result["ligand_based_score"].is_not_null().sum()
    print(f"Written: {output_path}  ({n_scored:,}/{len(result):,} scored)")


def main():
    parser = argparse.ArgumentParser(
        description="Ligand-based screening — standalone CLI (no SLURM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Path to input manifest.parquet")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path to write manifest_ligand_based.parquet")
    parser.add_argument("--work-dir", type=Path, default=Path("data/ligand_based"),
                        help="Root directory for per-target intermediates (default: data/ligand_based)")
    parser.add_argument("--method", default="usrcat",
                        help="Method registry key (default: usrcat)")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"),
                        help="Path to config.yaml (default: config/config.yaml)")
    parser.add_argument("--target", dest="targets", action="append",
                        help="Restrict to one or more targets (repeatable; omit for all)")
    parser.add_argument("--max-per-target", type=int, default=None,
                        help="Keep only the first N ligands per target (for quick testing)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config)).get("ligand_based", {})

    run(
        manifest_path=args.manifest,
        output_path=args.output,
        work_dir=args.work_dir,
        method_name=args.method,
        cfg=cfg,
        targets=args.targets,
        max_per_target=args.max_per_target,
    )


if __name__ == "__main__":
    main()
