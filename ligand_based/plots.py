#!/usr/bin/env python3
"""
make_plots.py (Refactored)
Creates publication-ready plots from evaluation results with improved parallelization.
"""
import os
import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterator
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve, auc

try:
    from rdkit.ML.Scoring import Scoring as RDScoring
except:
    RDScoring = None

plt.rcParams.update({"font.size": 11})
COLORS = {"Vina": "#3498db", "AEV-PLIG": "#e74c3c"}


# --------------------------
# Data structures
# --------------------------
@dataclass
class BootstrapTask:
    """Encapsulates parameters for a single bootstrap computation."""
    target: str
    method: str
    labels: np.ndarray
    scores: np.ndarray
    metric_type: str
    higher_is_better: bool
    n_bootstrap: int
    seed: int


@dataclass
class BootstrapResult:
    """Result from bootstrap computation."""
    target: str
    method: str
    values: List[float]


# --------------------------
# Helper functions
# --------------------------
def style_axes(ax):
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _ensure_binary(group, label_col="is_active"):
    return group[label_col].nunique() == 2


def _scores(group, col, higher_is_better):
    s = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=float)
    return s if higher_is_better else -s


# --------------------------
# Metric computation functions
# --------------------------
def compute_roc_auc(labels: np.ndarray, scores: np.ndarray, higher_is_better: bool) -> float:
    y_score = scores if higher_is_better else -scores
    fpr, tpr, _ = roc_curve(labels, y_score)
    return auc(fpr, tpr)


def compute_bedroc(labels: np.ndarray, scores: np.ndarray, higher_is_better: bool) -> float:
    df = pd.DataFrame({'y': labels.astype(bool), 's': scores.astype(float)})
    df = df.sort_values('s', ascending=not higher_is_better).reset_index(drop=True)
    arr = df[['s', 'y']].values.tolist()
    return float(RDScoring.CalcBEDROC(arr, 1, 20.0))


def compute_nef1(labels: np.ndarray, scores: np.ndarray, higher_is_better: bool) -> float:
    n = len(labels)
    df = pd.DataFrame({'y': labels.astype(bool), 's': scores.astype(float)})
    df = df.sort_values('s', ascending=not higher_is_better).reset_index(drop=True)
    arr = df[['s', 'y']].values.tolist()
    
    efs = RDScoring.CalcEnrichment(arr, 1, [0.01])
    ef1 = float(efs[0]) if isinstance(efs, (list, tuple, np.ndarray)) else float(efs)
    npos = int(labels.sum())
    top_n = max(1, math.ceil(n * 0.01))
    base = npos / n if n > 0 else 0
    if base > 0:
        ef_max = (min(npos, top_n) / top_n) / base
        return ef1 / ef_max if ef_max > 0 else np.nan
    return np.nan


def compute_ef1(labels: np.ndarray, scores: np.ndarray, higher_is_better: bool) -> float:
    df = pd.DataFrame({'y': labels.astype(bool), 's': scores.astype(float)})
    df = df.sort_values('s', ascending=not higher_is_better).reset_index(drop=True)
    arr = df[['s', 'y']].values.tolist()
    efs = RDScoring.CalcEnrichment(arr, 1, [0.01])
    return float(efs[0]) if isinstance(efs, (list, tuple, np.ndarray)) else float(efs)


METRIC_FUNCTIONS = {
    'ROC-AUC': compute_roc_auc,
    'BEDROC(20)': compute_bedroc,
    'NEF1%': compute_nef1,
    'EF1%': compute_ef1
}


# --------------------------
# Bootstrap computation
# --------------------------
def _compute_single_bootstrap(labels: np.ndarray, scores: np.ndarray, 
                             metric_func, higher_is_better: bool) -> float:
    """Compute metric for one bootstrap sample."""
    n = len(labels)
    indices = np.random.choice(n, size=n, replace=True)
    boot_labels = labels[indices]
    boot_scores = scores[indices]
    
    if len(np.unique(boot_labels)) < 2:
        return np.nan
    
    try:
        return metric_func(boot_labels, boot_scores, higher_is_better)
    except:
        return np.nan


def bootstrap_worker(task: BootstrapTask) -> BootstrapResult:
    """Worker function for parallel bootstrap computation."""
    # Validate data
    valid = ~np.isnan(task.scores)
    labels = task.labels[valid]
    scores = task.scores[valid]
    
    if len(labels) < 10 or len(np.unique(labels)) < 2:
        return BootstrapResult(task.target, task.method, [])
    
    # Get metric function
    metric_func = METRIC_FUNCTIONS.get(task.metric_type)
    if metric_func is None or (metric_func != compute_roc_auc and RDScoring is None):
        return BootstrapResult(task.target, task.method, [])
    
    # Set random seed for reproducibility
    np.random.seed(task.seed)
    
    # Compute bootstrap samples
    bootstrap_vals = []
    for _ in range(task.n_bootstrap):
        val = _compute_single_bootstrap(labels, scores, metric_func, task.higher_is_better)
        if np.isfinite(val):
            bootstrap_vals.append(val)
    
    return BootstrapResult(task.target, task.method, bootstrap_vals)


def generate_bootstrap_tasks(merged_df: pd.DataFrame, 
                            methods_spec: Dict,
                            metric_name: str,
                            n_boot: int) -> Iterator[BootstrapTask]:
    """Generator that yields bootstrap tasks for processing."""
    targets = merged_df['Protein_ID'].unique()
    
    for target in targets:
        target_data = merged_df[merged_df['Protein_ID'] == target]
        
        for method, spec in methods_spec.items():
            if method not in target_data.columns:
                continue
            
            clean_mask = target_data[method].notna() & target_data['is_active'].notna()
            if clean_mask.sum() < 10:
                continue
            
            labels = target_data.loc[clean_mask, 'is_active'].astype(int).values
            scores = target_data.loc[clean_mask, method].astype(float).values
            seed = hash((target, method, metric_name)) % (2**32)
            
            yield BootstrapTask(
                target=target,
                method=method,
                labels=labels,
                scores=scores,
                metric_type=metric_name,
                higher_is_better=spec["higher_is_better"],
                n_bootstrap=n_boot,
                seed=seed
            )


def compute_bootstrap_metrics_parallel(merged_df: pd.DataFrame,
                                       methods_spec: Dict,
                                       metric_name: str,
                                       n_boot: int,
                                       n_workers: int = None) -> Dict[str, Dict[str, List[float]]]:
    """Compute bootstrap metrics in parallel and return organized results."""
    if n_workers is None:
        n_workers = cpu_count()
    
    # Generate tasks
    tasks = list(generate_bootstrap_tasks(merged_df, methods_spec, metric_name, n_boot))
    
    if not tasks:
        return {}
    
    # Process in parallel with progress bar
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(bootstrap_worker, tasks, chunksize=max(1, len(tasks) // (n_workers * 4))),
            total=len(tasks),
            desc=f"    Bootstrap {metric_name}",
            leave=False
        ))
    
    # Organize results by target
    bootstrap_data = {}
    for result in results:
        if result.values:  # Only include non-empty results
            if result.target not in bootstrap_data:
                bootstrap_data[result.target] = {}
            bootstrap_data[result.target][result.method] = result.values
    
    return bootstrap_data


# --------------------------
# Plotting functions
# --------------------------
def plot_average_roc_curves(merged_df, methods_spec, outdir):
    """Create ROC plot with both methods."""
    print("\nCreating Average ROC Curves...")
    plt.figure(figsize=(8, 6))
    common_fpr = np.linspace(0, 1, 201)
    
    for method, spec in methods_spec.items():
        tprs, aucs = [], []
        groups = list(merged_df.groupby("Protein_ID", sort=False))
        
        for _, g in tqdm(groups, desc=f"  Computing {method} ROCs", leave=False):
            if not _ensure_binary(g) or method not in g.columns:
                continue
            
            g_clean = g.dropna(subset=[method, "is_active"])
            if len(g_clean) < 10:
                continue
            
            y_true = g_clean["is_active"].astype(int).to_numpy()
            y_score = _scores(g_clean, method, spec["higher_is_better"])
            if not np.isfinite(y_score).any():
                continue
            
            try:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                interp = np.interp(common_fpr, fpr, tpr, left=0.0, right=1.0)
                interp[0] = 0.0
                tprs.append(interp)
                aucs.append(auc(fpr, tpr))
            except Exception:
                continue
        
        if tprs:
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(aucs) if aucs else float("nan")
            plt.plot(common_fpr, mean_tpr, color=COLORS.get(method, None), lw=2,
                     label=f"{method} (AUC={mean_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Average ROC Curves")
    plt.legend(loc="lower right")
    style_axes(plt.gca())
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "average_ROC_curves.png"), dpi=300)
    plt.close()
    print("  ✓ Saved average_ROC_curves.png")


def plot_average_prc_curves(merged_df, methods_spec, outdir):
    """Create PRC plot with both methods."""
    print("\nCreating Average PRC Curves...")
    plt.figure(figsize=(8, 6))
    common_recall = np.linspace(0, 1, 201)
    
    for method, spec in methods_spec.items():
        precs, aucs = [], []
        groups = list(merged_df.groupby("Protein_ID", sort=False))
        
        for _, g in tqdm(groups, desc=f"  Computing {method} PRCs", leave=False):
            if not _ensure_binary(g) or method not in g.columns:
                continue
            
            g_clean = g.dropna(subset=[method, "is_active"])
            if len(g_clean) < 10:
                continue
            
            y_true = g_clean["is_active"].astype(int).to_numpy()
            y_score = _scores(g_clean, method, spec["higher_is_better"])
            if not np.isfinite(y_score).any():
                continue
            
            try:
                prec, rec, _ = precision_recall_curve(y_true, y_score)
                rec_inc_idx = np.argsort(rec)
                rec_sorted = rec[rec_inc_idx]
                prec_sorted = prec[rec_inc_idx]
                interp = np.interp(common_recall, rec_sorted, prec_sorted, 
                                 left=prec_sorted[0], right=prec_sorted[-1])
                precs.append(interp)
                aucs.append(auc(rec, prec))
            except Exception:
                continue
        
        if precs:
            mean_prec = np.mean(precs, axis=0)
            mean_auc = np.mean(aucs) if aucs else float("nan")
            plt.plot(common_recall, mean_prec, color=COLORS.get(method, None), lw=2,
                     label=f"{method} (AUPR={mean_auc:.3f})")
    
    baseline = merged_df["is_active"].mean() if len(merged_df) > 0 else 0.0
    plt.axhline(baseline, color='gray', linestyle='--', linewidth=1, 
                label=f'Random ({baseline:.3f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Average Precision-Recall Curves")
    plt.legend(loc="upper right")
    style_axes(plt.gca())
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "average_PRC_curves.png"), dpi=300)
    plt.close()
    print("  ✓ Saved average_PRC_curves.png")


def plot_single_violin(target: str, target_data: pd.DataFrame, 
                      bootstrap_data: Dict[str, List[float]],
                      methods_spec: Dict, metric_key: str, metric_name: str,
                      outdir: str):
    """Create a single violin plot for one target."""
    n_compounds = len(target_data)
    n_actives = int(target_data['is_active'].sum())
    
    all_bootstrap_data = []
    method_labels = []
    method_colors = []
    
    for method in methods_spec.keys():
        if method in bootstrap_data and len(bootstrap_data[method]) > 5:
            all_bootstrap_data.append(bootstrap_data[method])
            method_labels.append(method)
            method_colors.append(COLORS.get(method, "gray"))
    
    if not all_bootstrap_data:
        return
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    positions = np.arange(1, len(all_bootstrap_data) + 1)
    parts = ax.violinplot(all_bootstrap_data, positions=positions,
                         showmeans=True, showmedians=True, widths=0.6)
    
    for pc, color in zip(parts['bodies'], method_colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        pc.set_linewidth(1.2)
    
    if 'cmeans' in parts:
        parts['cmeans'].set_edgecolor('black')
        parts['cmeans'].set_linewidth(1.5)
    if 'cmedians' in parts:
        parts['cmedians'].set_edgecolor('darkred')
        parts['cmedians'].set_linewidth(1.5)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(method_labels, fontsize=11)
    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"{target}\n(n={n_compounds}, actives={n_actives})", 
                fontsize=13, fontweight='bold')
    
    style_axes(ax)
    
    if metric_name in ['ROC-AUC', 'BEDROC(20)', 'NEF1%']:
        ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    # Save with sanitized filename
    safe_target = target.replace('/', '_').replace('\\', '_')
    filename = f"{safe_target}_{metric_key}_violin.png"
    plt.savefig(os.path.join(outdir, filename), dpi=300, bbox_inches='tight')
    plt.close()


def plot_bootstrap_violins(merged_df: pd.DataFrame, methods_spec: Dict, 
                          outdir: str, n_boot: int = 200, n_workers: int = None):
    """Create bootstrap violin plots for all targets."""
    print("\nCreating bootstrap violin plots for all targets...")
    
    target_sizes = merged_df.groupby('Protein_ID').size().sort_values(ascending=False)
    all_targets = target_sizes.index.tolist()
    print(f"  Processing {len(all_targets)} targets")
    
    metrics_to_plot = {
        "BEDROC20": "BEDROC(20)",
        "NEF1pct": "NEF1%", 
        "ROC-AUC": "ROC-AUC",
        "EF1pct": "EF1%"
    }
    
    # Create subdirectory
    target_plot_dir = os.path.join(outdir, 'per_target_violins')
    os.makedirs(target_plot_dir, exist_ok=True)
    
    # Process each metric
    for metric_key, metric_name in tqdm(metrics_to_plot.items(), desc="  Creating violin plots"):
        print(f"\n  Processing {metric_name}...")
        
        # Compute bootstrap metrics in parallel
        bootstrap_data = compute_bootstrap_metrics_parallel(
            merged_df, methods_spec, metric_name, n_boot, n_workers
        )
        
        # Plot each target
        for target in tqdm(all_targets, desc=f"    Plotting {metric_name}", leave=False):
            if target in bootstrap_data:
                target_data = merged_df[merged_df['Protein_ID'] == target]
                plot_single_violin(
                    target, target_data, bootstrap_data[target],
                    methods_spec, metric_key, metric_name, target_plot_dir
                )
        
        print(f"    ✓ Saved {len(bootstrap_data)} plots for {metric_name}")


# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(description='Create plots from evaluation results')
    parser.add_argument('--merged', default='output/scores.csv', 
                       help='Merged data CSV')
    parser.add_argument('--outdir', default='output/plots', help='Output directory')
    parser.add_argument('--n-boot', type=int, default=200, 
                       help='Bootstrap iterations for violins')
    parser.add_argument('--n-workers', type=int, default=None,
                       help='Number of parallel workers (default: all CPUs)')
    args = parser.parse_args()
    
    # Load data
    if not os.path.exists(args.merged):
        raise SystemExit(f"Merged data not found: {args.merged}\nRun evaluate_metrics.py first!")
    
    print(f"Loading merged data: {args.merged}")
    merged_df = pd.read_csv(args.merged)
    
    if not {"Protein_ID", "is_active"}.issubset(set(merged_df.columns)):
        raise SystemExit("Merged CSV missing 'Protein_ID' and/or 'is_active' columns.")
    
    # Define methods
    methods_spec = {
        "Vina": {"higher_is_better": False}, 
        "AEV-PLIG": {"higher_is_better": True}
    }
    
    # Data quality check
    print("\nData Quality:")
    merged_df = merged_df.dropna(subset=['is_active', 'Protein_ID'])
    print(f"  Compounds: {len(merged_df):,}")
    print(f"  Targets: {merged_df['Protein_ID'].nunique()}")
    print(f"  Actives: {merged_df['is_active'].sum():,} ({100*merged_df['is_active'].mean():.1f}%)")
    
    for method in methods_spec.keys():
        if method in merged_df.columns:
            n_valid = merged_df[method].notna().sum()
            print(f"  {method}: {n_valid:,} valid ({100*n_valid/len(merged_df):.1f}%)")
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Create plots
    plot_average_roc_curves(merged_df, methods_spec, args.outdir)
    plot_average_prc_curves(merged_df, methods_spec, args.outdir)
    plot_bootstrap_violins(merged_df, methods_spec, args.outdir, 
                          n_boot=args.n_boot, n_workers=args.n_workers)
    
    print(f"\n{'='*70}")
    print(f"✓ All plots saved to {args.outdir}/")
    print(f"{'='*70}")
    print("  - average_ROC_curves.png")
    print("  - average_PRC_curves.png")
    print(f"  - per_target_violins/ (individual plots for {merged_df['Protein_ID'].nunique()} targets)")
    print("      * [TARGET]_BEDROC20_violin.png")
    print("      * [TARGET]_NEF1pct_violin.png")
    print("      * [TARGET]_ROC-AUC_violin.png")
    print("      * [TARGET]_EF1pct_violin.png")
    print("\nDone!")


if __name__ == "__main__":
    main()