#!/usr/bin/env python3
"""
evaluate_metrics.py
Computes per-target metrics and aggregates across targets.
Outputs:
  - per_target_detailed.csv: Metrics for each target individually with Mann-Whitney tests
  - per_target_summary.csv: Aggregated metrics with bootstrap CIs and statistical tests
  - merged_data.csv: Combined input data for plotting
"""
import os
import argparse
import math

import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score
from rdkit.ML.Scoring import Scoring as RDScoring

# --------------------------
# Data Loading
# --------------------------
def load_and_merge_data(vina_path: str, vina_col: str,
                       aevplig_path: str, aevplig_col: str,
                       id_col: str, label_col: str, target_col: str) -> pd.DataFrame:
    """Load and merge Vina and AEV-PLIG data."""
    print("Loading data...")
    
    # Load Vina file
    vina_df = pd.read_csv(vina_path)
    required_cols = [id_col, label_col, target_col]
    if not all(c in vina_df.columns for c in required_cols):
        raise ValueError(f"Vina file missing required columns: {required_cols}")
    
    # Subset to needed columns
    vina_subset = vina_df[[id_col, label_col, target_col, vina_col]].copy()
    vina_subset = vina_subset.rename(columns={vina_col: 'Vina'})
    
    # Load AEV-PLIG file
    aevplig_df = pd.read_csv(aevplig_path)
    if id_col not in aevplig_df.columns or aevplig_col not in aevplig_df.columns:
        raise ValueError(f"AEV-PLIG file missing {id_col} or {aevplig_col}")
    
    aevplig_subset = aevplig_df[[id_col, aevplig_col]].copy()
    aevplig_subset = aevplig_subset.rename(columns={aevplig_col: 'AEV-PLIG'})
    
    # Merge
    merged = vina_subset.merge(aevplig_subset, on=id_col, how='inner')
    
    # Filter out rows with missing critical data
    initial = len(merged)
    merged = merged.dropna(subset=[label_col, target_col])
    print(f"  Loaded {len(merged):,} compounds ({initial - len(merged)} removed with missing labels/targets)")
    print(f"  Targets: {merged[target_col].nunique()}")
    print(f"  Actives: {merged[label_col].sum():,} ({100*merged[label_col].median():.1f}%)")
    
    return merged

# --------------------------
# Metric computation
# --------------------------
def _rdkit_table(labels, scores, higher_is_better):
    """Prepare data for RDKit metrics."""
    df = pd.DataFrame({"y": labels.astype(bool), "s": scores.astype(float)})
    df = df.sort_values("s", ascending=not higher_is_better).reset_index(drop=True)
    return df[["s", "y"]].values.tolist()

def compute_metrics_for_target(target_df: pd.DataFrame, method_col: str, 
                                label_col: str, higher_is_better: bool,
                                fracs: list, bedroc_alpha: float) -> dict:
    """Compute all metrics for one target-method combination."""
    # Clean data
    clean = target_df.dropna(subset=[method_col, label_col]).copy()
    if len(clean) < 10:
        return None
    
    y = clean[label_col].astype(int).values
    scores = clean[method_col].astype(float).values
    
    if len(np.unique(y)) < 2:
        return None
    
    try:
        # Prepare for RDKit
        arr = _rdkit_table(y, scores, higher_is_better)
        
        # Compute metrics
        metrics = {}
        metrics['ROC-AUC'] = float(RDScoring.CalcAUC(arr, 1))
        metrics['BEDROC'] = float(RDScoring.CalcBEDROC(arr, 1, bedroc_alpha))
        
        # PR-AUC
        y_score = scores if higher_is_better else -scores
        metrics['PR-AUC'] = average_precision_score(y, y_score)
        
        # Enrichment factors
        for frac in fracs:
            efs = RDScoring.CalcEnrichment(arr, 1, [frac])
            ef = float(efs[0]) if isinstance(efs, (list, tuple, np.ndarray)) else float(efs)
            metrics[f'EF{int(frac*100)}%'] = ef
            
            # NEF
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
        print(f"    Error computing metrics: {e}")
        return None

# --------------------------
# Per-target evaluation
# --------------------------
def evaluate_per_target(df: pd.DataFrame, target_col: str, label_col: str,
                       fracs: list, bedroc_alpha: float) -> pd.DataFrame:
    """Compute metrics per target for both methods."""
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

# --------------------------
# Aggregation with bootstrap
# --------------------------
def aggregate_with_bootstrap(per_target_df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Aggregate per-target metrics with bootstrap CIs and Mann-Whitney U tests."""
    print("\nAggregating metrics across targets with bootstrap and statistical tests...")
    
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
            
            # median
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
        
        # Perform Mann-Whitney U test if both methods have values
        if 'Vina' in method_values and 'AEV-PLIG' in method_values:
            vina_vals = method_values['Vina']
            aevplig_vals = method_values['AEV-PLIG']
            
            # Mann-Whitney U test (two-sided)
            statistic, p_value = stats.mannwhitneyu(
                aevplig_vals, vina_vals, 
                alternative='two-sided'
            )
            
            row['p_value'] = round(p_value, 4)
            
            # Simplified significance
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

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(description='Evaluate virtual screening metrics per-target')
    parser.add_argument('--vina', default="../LIT_PCBA/aev_plig.csv:pK", 
                        help='Vina CSV path:column (e.g., file.csv:pK)')
    parser.add_argument('--aev-plig', default= "../AEV-PLIG/output/predictions/LIT_PCBA_predictions.csv:preds", 
                        help='AEV-PLIG CSV path:column')
    parser.add_argument('--id-col', default='unique_id', help='Compound ID column')
    parser.add_argument('--label-col', default='is_active', help='Label column')
    parser.add_argument('--target-col', default='Protein_ID', help='Target column')
    parser.add_argument('--outdir', default='results', help='Output directory')
    parser.add_argument('--fracs', default='0.01,0.05,0.1', help='Enrichment fractions')
    parser.add_argument('--bedroc-alpha', type=float, default=20.0, help='BEDROC alpha')
    parser.add_argument('--n-boot', type=int, default=1000, help='Bootstrap iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Parse file:column arguments
    vina_path, vina_col = args.vina.split(':') if ':' in args.vina else (args.vina, 'pK')
    aevplig_path, aevplig_col = args.aev_plig.split(':') if ':' in args.aev_plig else (args.aev_plig, 'preds')
    fracs = [float(x) for x in args.fracs.split(',')]
    
    # Load data
    df = load_and_merge_data(vina_path, vina_col, aevplig_path, aevplig_col,
                             args.id_col, args.label_col, args.target_col)
    
    # Evaluate per-target
    per_target_df = evaluate_per_target(df, args.target_col, args.label_col,
                                        fracs, args.bedroc_alpha)
    
    # Aggregate with bootstrap
    summary_df = aggregate_with_bootstrap(per_target_df, args.n_boot, args.seed)
    
    # Save outputs
    os.makedirs(args.outdir, exist_ok=True)
    
    detailed_path = os.path.join(args.outdir, 'per_target_detailed.csv')
    per_target_df.to_csv(detailed_path, index=False)
    print(f"\n✓ Saved detailed per-target metrics: {detailed_path}")
    
    summary_path = os.path.join(args.outdir, 'per_target_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved aggregated summary: {summary_path}")
    
    merged_path = os.path.join(args.outdir, 'merged_data.csv')
    df.to_csv(merged_path, index=False)
    print(f"✓ Saved merged data for plotting: {merged_path}")
    
    # Print summary
    print("\n" + "="*90)
    print("SUMMARY: Aggregated Metrics Across Targets")
    print("="*90)
    display_cols = ['Metric', 'Vina', 'AEV-PLIG', 'p_value', 'Significant', 'Effect_Size']
    print(summary_df[display_cols].to_string(index=False))
    print("="*90)
    print("\nSignificant: Yes if p<0.05, No if p≥0.05")
    print("Effect Size: Rank-biserial correlation (AEV-PLIG vs Vina)")
    print(f"\nEvaluated {len(per_target_df)} targets")
    print("Each compound evaluated within its assigned target.")
    print("Metrics averaged across targets with bootstrap 95% CIs.")
    print("Mann-Whitney U test used for between-method comparisons.")
    print("\nDone!")

if __name__ == '__main__':
    main()