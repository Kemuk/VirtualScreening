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
import sys
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
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

def load_manifest_data(manifest_path: Path) -> pd.DataFrame:
    """
    Load manifest and prepare for metrics computation.

    Args:
        manifest_path: Path to manifest parquet file

    Returns:
        DataFrame with required columns
    """
    print(f"Loading manifest from {manifest_path}...")
    manifest = pq.read_table(manifest_path).to_pandas()

    # Filter to rescored ligands
    rescored = manifest[manifest['rescoring_status'] == True].copy()
    print(f"  Total entries: {len(manifest)}")
    print(f"  Rescored entries: {len(rescored)}")

    if len(rescored) == 0:
        print("WARNING: No rescored ligands found. Using docked ligands instead.")
        rescored = manifest[manifest['docking_status'] == True].copy()

    # Prepare columns for metrics
    # Vina score: lower is better (more negative = stronger binding)
    # AEV-PLIG score: higher is better
    rescored['Vina'] = rescored['vina_score']
    rescored['AEV-PLIG'] = rescored['aev_plig_best_score']

    # Summary statistics
    print(f"\nData summary:")
    print(f"  Targets: {rescored['protein_id'].nunique()}")
    print(f"  Compounds: {len(rescored)}")
    print(f"  Actives: {rescored['is_active'].sum()} ({100*rescored['is_active'].mean():.1f}%)")

    for method in ['Vina', 'AEV-PLIG']:
        if method in rescored.columns:
            valid = rescored[method].notna().sum()
            print(f"  {method} valid: {valid} ({100*valid/len(rescored):.1f}%)")

    return rescored


# =============================================================================
# Metric Computation
# =============================================================================

def _rdkit_table(labels: np.ndarray, scores: np.ndarray, higher_is_better: bool) -> list:
    """Prepare data for RDKit metrics."""
    df = pd.DataFrame({"y": labels.astype(bool), "s": scores.astype(float)})
    df = df.sort_values("s", ascending=not higher_is_better).reset_index(drop=True)
    return df[["s", "y"]].values.tolist()


def compute_metrics_for_target(
    target_df: pd.DataFrame,
    method_col: str,
    label_col: str,
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
    clean = target_df.dropna(subset=[method_col, label_col]).copy()
    if len(clean) < 10:
        return None

    y = clean[label_col].astype(int).values
    scores = clean[method_col].astype(float).values

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


def evaluate_per_target(
    df: pd.DataFrame,
    target_col: str,
    label_col: str,
    fracs: List[float],
    bedroc_alpha: float,
) -> pd.DataFrame:
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

    methods = {
        'Vina': {'higher_is_better': False},
        'AEV-PLIG': {'higher_is_better': True}
    }

    results = []
    targets = df[target_col].unique()

    for target in tqdm(targets, desc="Processing targets"):
        target_df = df[df[target_col] == target]
        n_compounds = len(target_df)
        n_actives = int(target_df[label_col].sum())

        row = {
            'Target': target,
            'N_Compounds': n_compounds,
            'N_Actives': n_actives,
            'Active_Rate': n_actives / n_compounds if n_compounds > 0 else 0
        }

        for method, spec in methods.items():
            if method not in target_df.columns:
                continue

            metrics = compute_metrics_for_target(
                target_df, method, label_col,
                spec['higher_is_better'], fracs, bedroc_alpha
            )

            if metrics:
                for metric_name, value in metrics.items():
                    row[f'{method}_{metric_name}'] = value

        results.append(row)

    results_df = pd.DataFrame(results)
    print(f"  Computed metrics for {len(results_df)} targets")

    return results_df


def aggregate_with_bootstrap(
    per_target_df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
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
    metric_cols = [c for c in per_target_df.columns if c.startswith('Vina_')]
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

            values = per_target_df[col].dropna().values
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

    return pd.DataFrame(results)


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

    args = parser.parse_args()

    # Parse fractions
    fracs = [float(x) for x in args.fracs.split(',')]

    # Validate inputs
    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    # Load data
    df = load_manifest_data(args.manifest)

    if len(df) == 0:
        print("ERROR: No data to process", file=sys.stderr)
        sys.exit(1)

    # Compute per-target metrics
    per_target_df = evaluate_per_target(
        df,
        target_col='protein_id',
        label_col='is_active',
        fracs=fracs,
        bedroc_alpha=args.bedroc_alpha,
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
    per_target_df.to_csv(per_target_path, index=False)
    print(f"\nSaved per-target metrics: {per_target_path}")

    summary_path = args.outdir / 'summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Print summary
    print("\n" + "="*90)
    print("SUMMARY: Aggregated Metrics Across Targets")
    print("="*90)

    display_cols = ['Metric', 'Vina', 'AEV-PLIG', 'p_value', 'Significant', 'Effect_Size']
    display_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[display_cols].to_string(index=False))

    print("="*90)
    print(f"\nEvaluated {len(per_target_df)} targets")
    print("Significant: Yes if p<0.05")
    print("Effect Size: Rank-biserial correlation (positive = AEV-PLIG better)")

    print("\nResults computation complete!")
    sys.exit(0)


if __name__ == '__main__':
    main()
