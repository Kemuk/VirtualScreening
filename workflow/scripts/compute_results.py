#!/usr/bin/env python3
"""
compute_results.py

Compute virtual screening metrics from the manifest parquet file.

Reads manifest with docking and rescoring results, computes:
  - Per-target metrics: ROC-AUC, BEDROC, PR-AUC, EF, NEF
  - Aggregated metrics with bootstrap confidence intervals
  - Statistical comparisons (Mann-Whitney U tests)

Outputs:
  - per_target_metrics.csv: Metrics for each target
  - summary.csv: Aggregated metrics with CIs and p-values
"""

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl
from scipy import stats
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score

try:
    from rdkit.ML.Scoring import Scoring as RDScoring
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("WARNING: RDKit not available. Some metrics will be skipped.", file=sys.stderr)


# =============================================================================
# Data Loading
# =============================================================================

def load_manifest_data(manifest_path: Path) -> pl.DataFrame:
    """
    Load manifest and prepare for metrics computation.

    Args:
        manifest_path: Path to manifest parquet file

    Returns:
        DataFrame with required columns
    """
    print(f"Loading manifest from {manifest_path}...")
    manifest = pl.read_parquet(manifest_path)

    # Filter to rescored ligands
    rescored = manifest.filter(pl.col("rescoring_status") == True)
    print(f"  Total entries: {manifest.height}")
    print(f"  Rescored entries: {rescored.height}")

    if rescored.is_empty():
        print("WARNING: No rescored ligands found. Using docked ligands instead.")
        rescored = manifest.filter(pl.col("docking_status") == True)

    # Prepare columns for metrics
    # Vina score: lower is better (more negative = stronger binding)
    # AEV-PLIG score: higher is better
    rescored = rescored.with_columns(
        pl.col("vina_score").alias("Vina"),
        pl.col("aev_plig_best_score").alias("AEV-PLIG"),
    )

    # Summary statistics
    print(f"\nData summary:")
    targets = rescored.select(pl.col("protein_id").n_unique()).item()
    actives = rescored.select(pl.col("is_active").sum()).item()
    active_rate = rescored.select(pl.col("is_active").mean()).item()
    print(f"  Targets: {targets}")
    print(f"  Compounds: {rescored.height}")
    print(f"  Actives: {actives} ({100*active_rate:.1f}%)")

    for method in ['Vina', 'AEV-PLIG']:
        if method in rescored.columns:
            valid = rescored.select(pl.col(method).is_not_null().sum()).item()
            print(f"  {method} valid: {valid} ({100*valid/rescored.height:.1f}%)")

    return rescored


# =============================================================================
# Metric Computation
# =============================================================================

def _rdkit_table(labels: np.ndarray, scores: np.ndarray, higher_is_better: bool) -> list:
    """Prepare data for RDKit metrics."""
    order = np.argsort(scores)
    if higher_is_better:
        order = order[::-1]
    sorted_scores = scores[order].astype(float)
    sorted_labels = labels[order].astype(bool)
    return list(map(list, zip(sorted_scores, sorted_labels)))


def compute_metrics_for_target(
    labels: np.ndarray,
    scores: np.ndarray,
    higher_is_better: bool,
    fracs: List[float],
    bedroc_alpha: float,
) -> Optional[Dict]:
    """
    Compute all metrics for one target-method combination.

    Args:
        target_df: DataFrame with target data
        method_col: Column name for scores
        label_col: Column name for labels
        higher_is_better: Whether higher scores are better
        fracs: Enrichment fractions to compute
        bedroc_alpha: BEDROC alpha parameter

    Returns:
        Dictionary with metrics or None if failed
    """
    # Clean data
    valid_mask = np.isfinite(scores) & np.isfinite(labels)
    labels = labels[valid_mask]
    scores = scores[valid_mask]

    if len(labels) < 10:
        return None

    y = labels.astype(int)
    scores = scores.astype(float)

    if len(np.unique(y)) < 2:
        return None

    try:
        metrics = {}

        # ROC-AUC using sklearn (works without RDKit)
        from sklearn.metrics import roc_auc_score
        y_score = scores if higher_is_better else -scores
        metrics['ROC-AUC'] = roc_auc_score(y, y_score)

        # PR-AUC
        metrics['PR-AUC'] = average_precision_score(y, y_score)

        # RDKit-based metrics
        if HAS_RDKIT:
            arr = _rdkit_table(y, scores, higher_is_better)

            # BEDROC
            metrics['BEDROC'] = float(RDScoring.CalcBEDROC(arr, 1, bedroc_alpha))

            # Enrichment factors
            for frac in fracs:
                efs = RDScoring.CalcEnrichment(arr, 1, [frac])
                ef = float(efs[0]) if isinstance(efs, (list, tuple, np.ndarray)) else float(efs)
                metrics[f'EF{int(frac*100)}%'] = ef

                # NEF (Normalized Enrichment Factor)
                n_pos = int(y.sum())
                top_n = max(1, math.ceil(len(y) * frac))
                base_rate = n_pos / len(y) if len(y) > 0 else 0
                if base_rate > 0:
                    ef_max = (min(n_pos, top_n) / top_n) / base_rate
                    nef = ef / ef_max if ef_max > 0 else 0
                    metrics[f'NEF{int(frac*100)}%'] = float(np.clip(nef, 0, 1))
                else:
                    metrics[f'NEF{int(frac*100)}%'] = np.nan

        return metrics

    except Exception as e:
        print(f"    Error computing metrics: {e}", file=sys.stderr)
        return None


def _compute_target_metrics(
    target: str,
    target_data: Dict[str, np.ndarray],
    label_col: str,
    fracs: List[float],
    bedroc_alpha: float,
) -> Optional[Dict]:
    labels = target_data.get(label_col)
    if labels is None:
        return None

    labels = labels.astype(int)
    n_compounds = len(labels)
    n_actives = int(labels.sum())

    row = {
        "Target": target,
        "N_Compounds": n_compounds,
        "N_Actives": n_actives,
        "Active_Rate": n_actives / n_compounds if n_compounds > 0 else 0,
    }

    methods = {
        "Vina": {"higher_is_better": False},
        "AEV-PLIG": {"higher_is_better": True},
    }

    for method, spec in methods.items():
        scores = target_data.get(method)
        if scores is None:
            continue

        metrics = compute_metrics_for_target(
            labels,
            scores,
            spec["higher_is_better"],
            fracs,
            bedroc_alpha,
        )

        if metrics:
            for metric_name, value in metrics.items():
                row[f"{method}_{metric_name}"] = value

    return row


def evaluate_per_target(
    df: pl.DataFrame,
    target_col: str,
    label_col: str,
    fracs: List[float],
    bedroc_alpha: float,
    n_jobs: int,
) -> pl.DataFrame:
    """
    Compute metrics per target for both methods.

    Args:
        df: DataFrame with all data
        target_col: Column name for target ID
        label_col: Column name for labels
        fracs: Enrichment fractions
        bedroc_alpha: BEDROC alpha

    Returns:
        DataFrame with per-target metrics
    """
    print("\nComputing per-target metrics...")

    results = []
    target_groups = df.partition_by(target_col, as_dict=True)

    def build_target_payload(target_df: pl.DataFrame) -> Dict[str, np.ndarray]:
        return {
            label_col: target_df.get_column(label_col).to_numpy(),
            "Vina": target_df.get_column("Vina").to_numpy() if "Vina" in target_df.columns else None,
            "AEV-PLIG": target_df.get_column("AEV-PLIG").to_numpy() if "AEV-PLIG" in target_df.columns else None,
        }

    items = []
    for target, target_df in target_groups.items():
        items.append((target, build_target_payload(target_df)))

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                _compute_target_metrics,
                target,
                payload,
                label_col,
                fracs,
                bedroc_alpha,
            )
            for target, payload in items
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing targets"):
            result = future.result()
            if result:
                results.append(result)

    results_df = pl.DataFrame(results)
    print(f"  Computed metrics for {results_df.height} targets")

    return results_df


def aggregate_with_bootstrap(
    per_target_df: pl.DataFrame,
    n_boot: int,
    seed: int,
) -> pl.DataFrame:
    """
    Aggregate per-target metrics with bootstrap CIs and Mann-Whitney U tests.

    Args:
        per_target_df: DataFrame with per-target metrics
        n_boot: Number of bootstrap iterations
        seed: Random seed

    Returns:
        DataFrame with aggregated metrics
    """
    print("\nAggregating metrics with bootstrap CIs...")

    methods = ['Vina', 'AEV-PLIG']

    # Get metric names
    metric_cols = [c for c in per_target_df.columns if c.startswith("Vina_")]
    metric_names = [c.replace('Vina_', '') for c in metric_cols]

    results = []
    rng = np.random.default_rng(seed)

    for metric_name in tqdm(metric_names, desc="Aggregating metrics"):
        row = {'Metric': metric_name}

        # Collect values for both methods
        method_values = {}

        for method in methods:
            col = f'{method}_{metric_name}'
            if col not in per_target_df.columns:
                continue

            values = per_target_df.get_column(col).drop_nulls().to_numpy()
            if len(values) == 0:
                continue

            method_values[method] = values

            # Median
            median_val = np.median(values)

            # Bootstrap CI
            boot_vals = []
            for _ in range(n_boot):
                boot_sample = rng.choice(values, size=len(values), replace=True)
                boot_vals.append(np.median(boot_sample))

            lo, hi = np.percentile(boot_vals, [2.5, 97.5])

            row[method] = f"{median_val:.3f} [{lo:.3f}-{hi:.3f}]"
            row[f'{method}_median'] = round(median_val, 3)
            row[f'{method}_lo'] = round(lo, 3)
            row[f'{method}_hi'] = round(hi, 3)

        # Mann-Whitney U test
        if 'Vina' in method_values and 'AEV-PLIG' in method_values:
            vina_vals = method_values['Vina']
            aevplig_vals = method_values['AEV-PLIG']

            statistic, p_value = stats.mannwhitneyu(
                aevplig_vals, vina_vals,
                alternative='two-sided'
            )

            row['p_value'] = round(p_value, 4)
            row['Significant'] = 'Yes' if p_value < 0.05 else 'No'

            # Effect size: rank-biserial correlation
            n1, n2 = len(aevplig_vals), len(vina_vals)
            r = 1 - (2*statistic) / (n1 * n2)
            row['Effect_Size'] = round(r, 3)
        else:
            row['p_value'] = np.nan
            row['Significant'] = 'N/A'
            row['Effect_Size'] = np.nan

        results.append(row)

    return pl.DataFrame(results)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute virtual screening metrics from manifest'
    )
    parser.add_argument(
        '--manifest',
        type=Path,
        required=True,
        help='Path to manifest parquet file'
    )
    parser.add_argument(
        '--outdir',
        type=Path,
        default=Path('results'),
        help='Output directory'
    )
    parser.add_argument(
        '--fracs',
        type=str,
        default='0.01,0.05,0.10',
        help='Enrichment fractions (comma-separated)'
    )
    parser.add_argument(
        '--bedroc-alpha',
        type=float,
        default=20.0,
        help='BEDROC alpha parameter'
    )
    parser.add_argument(
        '--n-boot',
        type=int,
        default=1000,
        help='Bootstrap iterations'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=max(os.cpu_count() or 1, 1),
        help='Number of parallel workers for per-target computation'
    )

    args = parser.parse_args()

    # Parse fractions
    fracs = [float(x) for x in args.fracs.split(',')]

    # Validate inputs
    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    # Load data
    df = load_manifest_data(args.manifest)

    if df.is_empty():
        print("ERROR: No data to process", file=sys.stderr)
        sys.exit(1)

    # Compute per-target metrics
    per_target_df = evaluate_per_target(
        df,
        target_col='protein_id',
        label_col='is_active',
        fracs=fracs,
        bedroc_alpha=args.bedroc_alpha,
        n_jobs=args.n_jobs,
    )

    # Aggregate with bootstrap
    summary_df = aggregate_with_bootstrap(
        per_target_df,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    # Save outputs
    args.outdir.mkdir(parents=True, exist_ok=True)

    per_target_path = args.outdir / 'per_target_metrics.csv'
    per_target_df.write_csv(per_target_path)
    print(f"\nSaved per-target metrics: {per_target_path}")

    summary_path = args.outdir / 'summary.csv'
    summary_df.write_csv(summary_path)
    print(f"Saved summary: {summary_path}")

    # Print summary
    print("\n" + "="*90)
    print("SUMMARY: Aggregated Metrics Across Targets")
    print("="*90)

    display_cols = ['Metric', 'Vina', 'AEV-PLIG', 'p_value', 'Significant', 'Effect_Size']
    display_cols = [c for c in display_cols if c in summary_df.columns]
    display_rows = summary_df.select(display_cols).to_dicts()
    if display_rows:
        header = " ".join(f"{col:>15}" for col in display_cols)
        print(header)
        for row in display_rows:
            print(" ".join(f"{str(row.get(col, '')):>15}" for col in display_cols))

    print("="*90)
    print(f"\nEvaluated {per_target_df.height} targets")
    print("Significant: Yes if p<0.05")
    print("Effect Size: Rank-biserial correlation (positive = AEV-PLIG better)")

    print("\nResults computation complete!")
    sys.exit(0)


# =============================================================================
# Batch Processing (for SLURM array jobs)
# =============================================================================

def process_batch(items: list, config: dict) -> list:
    """
    Process a batch of items for results computation.

    Called by the SLURM worker to process a chunk of items.
    For results computation, this is typically done per-target.

    Args:
        items: List of item records from manifest (dicts with ligand info)
        config: Workflow configuration dict

    Returns:
        List of result records with 'ligand_id', 'success', 'data'
    """
    results = []

    # Group items by protein_id for per-target computation
    from collections import defaultdict
    by_target = defaultdict(list)
    for item in items:
        by_target[item['protein_id']].append(item)

    fracs = config.get('results', {}).get('enrichment_fractions', [0.01, 0.05, 0.10])
    bedroc_alpha = config.get('results', {}).get('bedroc_alpha', 20.0)

    for protein_id, target_items in by_target.items():
        ligand_id = target_items[0]['ligand_id']  # Use first ligand as reference

        try:
            # Convert items to DataFrame for metric computation
            target_df = pl.DataFrame(target_items)

            # Check if we have enough data
            if target_df.height < 2:
                results.append({
                    'ligand_id': ligand_id,
                    'success': False,
                    'error': f'Insufficient data for target {protein_id}',
                })
                continue

            # Prepare columns
            target_df = target_df.with_columns(
                pl.col("vina_score").alias("Vina"),
                pl.col("aev_plig_best_score").alias("AEV-PLIG"),
            )

            # Compute metrics for this target
            metrics = {}
            for method, higher_is_better in [('Vina', False), ('AEV-PLIG', True)]:
                if method not in target_df.columns:
                    continue
                valid = target_df.filter(pl.col(method).is_not_null())
                if valid.height < 2:
                    continue

                try:
                    labels = valid.get_column("is_active").to_numpy()
                    scores = valid.get_column(method).to_numpy()
                    method_metrics = compute_metrics_for_target(
                        labels=labels,
                        scores=scores,
                        higher_is_better=higher_is_better,
                        fracs=fracs,
                        bedroc_alpha=bedroc_alpha,
                    )
                    metrics[method] = method_metrics
                except Exception as e:
                    print(f"WARNING: Failed to compute {method} metrics for {protein_id}: {e}")

            results.append({
                'ligand_id': ligand_id,
                'success': True,
                'data': {
                    'protein_id': protein_id,
                        'n_compounds': target_df.height,
                        'n_actives': int(target_df.select(pl.col("is_active").sum()).item()),
                    'metrics': metrics,
                },
            })

        except Exception as e:
            results.append({
                'ligand_id': ligand_id,
                'success': False,
                'error': str(e),
            })

    return results


if __name__ == '__main__':
    main()
