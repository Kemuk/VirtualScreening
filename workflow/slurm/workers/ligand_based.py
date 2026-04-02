"""
ligand_based.py

Worker for ligand-based virtual screening (SMILES-only, no docking required).

Reads the manifest, scores every ligand per target against a template molecule
using the configured method (default: USRCAT shape similarity), and writes the
results to a separate output manifest with three new columns:

  ligand_based_score   float32  similarity to template (higher = more similar)
  ligand_based_status  bool     True when scoring succeeded
  ligand_based_method  str      name of the method used

Usage
-----
All targets (production / local):
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
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from workflow.ligand_based.registry import get_method


def run(manifest_path, output_path, work_dir, method_name, cfg, targets=None):
    """
    Score all ligands in the manifest and write to output_path.

    Args:
        manifest_path: Path to input manifest.parquet
        output_path:   Path to write manifest_ligand_based.parquet
        work_dir:      Root for per-target intermediates (conformers, vectors)
        method_name:   Registry key for the LigandBasedMethod to use
        cfg:           ligand_based config dict (from config.yaml)
        targets:       Optional list of protein_id values to restrict processing
    """
    df = pd.read_parquet(manifest_path)

    if targets:
        df = df[df["protein_id"].isin(targets)].copy()

    # Initialise output columns
    df["ligand_based_score"] = float("nan")
    df["ligand_based_status"] = False
    df["ligand_based_method"] = method_name

    method = get_method(method_name)

    for protein_id, group in tqdm(df.groupby("protein_id"), desc="targets"):
        target_dir = Path(work_dir) / protein_id

        smiles = group["smiles_canonical"].tolist()
        actives = group.loc[group["is_active"], "smiles_canonical"]
        template = actives.iloc[0] if len(actives) else smiles[0]

        scores = method.run_target(smiles, template, target_dir, cfg)

        df.loc[group.index, "ligand_based_score"] = scores.values
        df.loc[group.index, "ligand_based_status"] = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    n_scored = int(df["ligand_based_status"].sum())
    print(f"Written: {output_path}  ({n_scored} / {len(df)} scored)")


def main():
    parser = argparse.ArgumentParser(
        description="Ligand-based screening worker — scores manifest ligands by shape similarity",
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
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config)).get("ligand_based", {})

    run(
        manifest_path=args.manifest,
        output_path=args.output,
        work_dir=args.work_dir,
        method_name=args.method,
        cfg=cfg,
        targets=args.targets,
    )


if __name__ == "__main__":
    main()
