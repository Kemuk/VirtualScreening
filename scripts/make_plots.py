#!/usr/bin/env python3
"""
make_plots.py - Optimized version
Creates publication-ready plots from evaluation results with efficient parallel processing.
Outputs:
  - average_ROC_curves.png
  - average_PRC_curves.png
  - per_target_curves/ (directory with individual plots)
  - BEDROC20_violin.png
  - NEF1pct_violin.png
  - ROC-AUC_violin.png
  - EF1pct_violin.png
"""
import os
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve, auc

try:
    from rdkit.ML.Scoring import Scoring as RDScoring
except ImportError:
    RDScoring = None

plt.rcParams.update({"font.size": 11})
COLORS = {"Vina": "#3498db", "AEV-PLIG": "#e74c3c"}

def style_axes(ax):
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --------------------------
# Unified curve computation
# --------------------------
def compute_curves_worker(args):
    """Compute BOTH ROC and PRC curves for one target-method combination."""
    target, method, labels, scores, higher_is_better = args
    
    labels = np.array(labels)
    scores = np.array(scores)
    
    # Remove NaN
    valid = ~np.isnan(scores)
    labels, scores = labels[valid], scores[valid]
    
    if len(labels) < 10 or len(np.unique(labels)) < 2:
        return None
    
    # Calculate per-target baseline (active ratio)
    baseline = np.mean(labels)
    
    try:
        y_score = scores if higher_is_better else -scores
        
        # ROC
        fpr, tpr, _ = roc_curve(labels, y_score)
        roc_auc = auc(fpr, tpr)
        
        # PRC
        prec, rec, _ = precision_recall_curve(labels, y_score)
        pr_auc = auc(rec, prec)
        
        return {
            'target': target,
            'method': method,
            'roc': (fpr, tpr, roc_auc),
            'prc': (prec, rec, pr_auc),
            'baseline': baseline  # Return baseline for individual PRC plots
        }
    except:
        return None

# --------------------------
# Unified bootstrap computation
# --------------------------
def compute_all_metrics_bootstrap_worker(args):
    """Compute ALL metrics with bootstrap for one target-method combination."""
    target, method, labels, scores, higher_is_better, n_bootstrap, seed = args
    
    labels = np.array(labels)
    scores = np.array(scores)
    
    # Remove NaN
    valid = ~np.isnan(scores)
    labels, scores = labels[valid], scores[valid]
    
    if len(labels) < 10 or len(np.unique(labels)) < 2:
        return None
    
    # Storage for all metrics
    results = {
        'target': target,
        'method': method,
        'ROC-AUC': [],
        'BEDROC(20)': [],
        'NEF1%': [],
        'EF1%': []
    }
    
    n = len(labels)
    rng = np.random.default_rng(seed)
    
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        boot_labels, boot_scores = labels[indices], scores[indices]
        
        if len(np.unique(boot_labels)) < 2:
            continue
        
        # ROC-AUC
        try:
            y_score = boot_scores if higher_is_better else -boot_scores
            fpr, tpr, _ = roc_curve(boot_labels, y_score)
            val = auc(fpr, tpr)
            if np.isfinite(val):
                results['ROC-AUC'].append(val)
        except:
            pass
        
        # RDKit-based metrics
        if RDScoring:
            try:
                df = pd.DataFrame({'y': boot_labels.astype(bool), 's': boot_scores})
                df = df.sort_values('s', ascending=not higher_is_better).reset_index(drop=True)
                arr = df[['s', 'y']].values.tolist()
                
                # BEDROC(20)
                bedroc_val = float(RDScoring.CalcBEDROC(arr, 1, 20.0))
                if np.isfinite(bedroc_val):
                    results['BEDROC(20)'].append(bedroc_val)
                
                # EF1% and NEF1%
                efs = RDScoring.CalcEnrichment(arr, 1, [0.01])
                ef1 = float(efs[0]) if isinstance(efs, (list, tuple, np.ndarray)) else float(efs)
                
                if np.isfinite(ef1):
                    results['EF1%'].append(ef1)
                    
                    # NEF1%
                    npos = int(boot_labels.sum())
                    top_n = max(1, math.ceil(len(boot_labels) * 0.01))
                    base = npos / len(boot_labels)
                    ef_max = (min(npos, top_n) / top_n) / base if base > 0 else 0
                    nef_val = ef1 / ef_max if ef_max > 0 else np.nan
                    if np.isfinite(nef_val):
                        results['NEF1%'].append(nef_val)
            except:
                pass
    
    return results

# --------------------------
# Plot functions
# --------------------------
def plot_average_curves(merged_df, methods_spec, outdir, max_workers=None):
    """Create ROC and PRC plots in one parallel pass."""
    print("\nCreating Average ROC and PRC Curves...")
    
    # Prepare all tasks
    tasks = []
    for method, spec in methods_spec.items():
        if method not in merged_df.columns:
            continue
        
        for target, group in merged_df.groupby("Protein_ID", sort=False):
            clean = group.dropna(subset=[method, "is_active"])
            if len(clean) < 10 or clean["is_active"].nunique() < 2:
                continue
            
            labels = clean["is_active"].astype(int).values
            scores = clean[method].astype(float).values
            tasks.append((target, method, labels, scores, spec["higher_is_better"]))
    
    # Parallel computation
    roc_data = {m: {'tprs': [], 'aucs': []} for m in methods_spec.keys()}
    prc_data = {m: {'precs': [], 'aucs': []} for m in methods_spec.keys()}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_curves_worker, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures),
                         desc="  Computing curves", leave=False):
            result = future.result()
            if result:
                method = result['method']
                
                # ROC data
                fpr, tpr, roc_auc = result['roc']
                common_fpr = np.linspace(0, 1, 201)
                interp = np.interp(common_fpr, fpr, tpr, left=0.0, right=1.0)
                interp[0] = 0.0
                roc_data[method]['tprs'].append(interp)
                roc_data[method]['aucs'].append(roc_auc)
                
                # PRC data
                prec, rec, pr_auc = result['prc']
                common_recall = np.linspace(0, 1, 201)
                sort_idx = np.argsort(rec)
                rec_sorted, prec_sorted = rec[sort_idx], prec[sort_idx]
                interp = np.interp(common_recall, rec_sorted, prec_sorted,
                                 left=prec_sorted[0], right=prec_sorted[-1])
                prc_data[method]['precs'].append(interp)
                prc_data[method]['aucs'].append(pr_auc)
    
    # Plot ROC
    plt.figure(figsize=(8, 6))
    common_fpr = np.linspace(0, 1, 201)
    for method in methods_spec.keys():
        if roc_data[method]['tprs']:
            mean_tpr = np.mean(roc_data[method]['tprs'], axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(roc_data[method]['aucs'])
            plt.plot(common_fpr, mean_tpr, color=COLORS.get(method), lw=2,
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
    
    # Plot PRC
    plt.figure(figsize=(8, 6))
    common_recall = np.linspace(0, 1, 201)
    for method in methods_spec.keys():
        if prc_data[method]['precs']:
            mean_prec = np.mean(prc_data[method]['precs'], axis=0)
            mean_auc = np.mean(prc_data[method]['aucs'])
            plt.plot(common_recall, mean_prec, color=COLORS.get(method), lw=2,
                     label=f"{method} (AUPR={mean_auc:.3f})")
    
    baseline = merged_df["is_active"].mean()
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

# --------------------------
# NEW FUNCTION: Per-Target Curves
# --------------------------
def plot_target_curves(merged_df, methods_spec, outdir, max_workers=None):
    """Create individual ROC and PRC plots for EACH target."""
    print("\nCreating per-target ROC and PRC Curves...")
    
    # Create a dedicated subdirectory for these plots
    target_plot_dir = os.path.join(outdir, "per_target_curves")
    os.makedirs(target_plot_dir, exist_ok=True)
    
    # --- 1. Prepare all tasks (copied from plot_average_curves) ---
    tasks = []
    for method, spec in methods_spec.items():
        if method not in merged_df.columns:
            continue
        
        for target, group in merged_df.groupby("Protein_ID", sort=False):
            clean = group.dropna(subset=[method, "is_active"])
            if len(clean) < 10 or clean["is_active"].nunique() < 2:
                continue
            
            labels = clean["is_active"].astype(int).values
            scores = clean[method].astype(float).values
            tasks.append((target, method, labels, scores, spec["higher_is_better"]))
    
    # --- 2. Parallel computation ---
    # We will collect all results and group them by target
    results_by_target = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_curves_worker, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures),
                         desc="  Computing all target curves", leave=False):
            result = future.result()
            if result:
                target = result['target']
                if target not in results_by_target:
                    results_by_target[target] = []
                # Append the full result dict (roc, prc, baseline, etc.)
                results_by_target[target].append(result)
    
    # --- 3. Plotting (one plot per target) ---
    print(f"  Saving {len(results_by_target)} per-target plots to {target_plot_dir}...")
    for target, results_list in tqdm(results_by_target.items(), 
                                   desc="  Saving plots", leave=False):
        
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        fig_prc, ax_prc = plt.subplots(figsize=(7, 6))
        target_baseline = None
        
        # Plot curves for all methods on the same axes
        for result in results_list:
            method = result['method']
            fpr, tpr, roc_auc = result['roc']
            prec, rec, pr_auc = result['prc']
            
            # This will be the same for all methods on this target
            target_baseline = result['baseline'] 
            
            # Plot ROC
            ax_roc.plot(fpr, tpr, color=COLORS.get(method), lw=2,
                         label=f"{method} (AUC={roc_auc:.3f})")
            
            # Plot PRC
            sort_idx = np.argsort(rec)
            rec_sorted, prec_sorted = rec[sort_idx], prec[sort_idx]
            ax_prc.plot(rec_sorted, prec_sorted, color=COLORS.get(method), lw=2,
                         label=f"{method} (AUPR={pr_auc:.3f})")

        # Finalize ROC plot
        ax_roc.plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Random')
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"ROC Curve - Target: {target}")
        ax_roc.legend(loc="lower right")
        style_axes(ax_roc)
        fig_roc.tight_layout()
        # Use lower DPI to avoid massive file sizes for many plots
        fig_roc.savefig(os.path.join(target_plot_dir, f"{target}_ROC.png"), dpi=150)
        plt.close(fig_roc)
        
        # Finalize PRC plot
        if target_baseline is not None:
            ax_prc.axhline(target_baseline, color='gray', linestyle='--', linewidth=1,
                           label=f'Random ({target_baseline:.3f})')
        ax_prc.set_xlabel("Recall")
        ax_prc.set_ylabel("Precision")
        ax_prc.set_title(f"Precision-Recall Curve - Target: {target}")
        ax_prc.legend(loc="upper right")
        style_axes(ax_prc)
        fig_prc.tight_layout()
        fig_prc.savefig(os.path.join(target_plot_dir, f"{target}_PRC.png"), dpi=150)
        plt.close(fig_prc)
    
    print("  ✓ Finished per-target plots.")

def plot_bootstrap_violins(merged_df, methods_spec, outdir, n_boot=200, max_workers=None):
    """Create bootstrap violin plots - compute all metrics in parallel."""
    print("\nCreating bootstrap violin plots...")
    
    # Get top targets
    target_sizes = merged_df.groupby('Protein_ID').size().sort_values(ascending=False)
    top_targets = target_sizes.head(8).index.tolist()
    print(f"  Using top {len(top_targets)} targets")
    
    # Prepare all tasks (one task per target-method combination)
    tasks = []
    for target in top_targets:
        target_data = merged_df[merged_df['Protein_ID'] == target]
        
        for method, spec in methods_spec.items():
            if method not in target_data.columns:
                continue
            
            clean = target_data.dropna(subset=[method, 'is_active'])
            if len(clean) < 10:
                continue
            
            labels = clean['is_active'].astype(int).values
            scores = clean[method].astype(float).values
            seed = hash((target, method)) % (2**32)
            
            tasks.append((target, method, labels, scores, 
                        spec["higher_is_better"], n_boot, seed))
    
    # Parallel computation - all metrics at once
    all_results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_all_metrics_bootstrap_worker, task): task 
                  for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures),
                         desc="  Computing all metrics with bootstrap", leave=False):
            result = future.result()
            if result:
                target = result['target']
                method = result['method']
                
                if target not in all_results:
                    all_results[target] = {}
                all_results[target][method] = result
    
    # Create plots for each metric
    metrics_to_plot = {
        "BEDROC20_violin.png": "BEDROC(20)",
        "NEF1pct_violin.png": "NEF1%",
        "ROC-AUC_violin.png": "ROC-AUC",
        "EF1pct_violin.png": "EF1%"
    }
    
    for filename, metric_name in metrics_to_plot.items():
        print(f"  Creating {filename}...")
        
        fig, axes = plt.subplots(len(top_targets), 1, figsize=(10, 2*len(top_targets)))
        if len(top_targets) == 1:
            axes = [axes]
        
        for idx, target in enumerate(top_targets):
            ax = axes[idx]
            target_data = merged_df[merged_df['Protein_ID'] == target]
            n_compounds = len(target_data)
            n_actives = int(target_data['is_active'].sum())
            
            # Collect bootstrap data for this target and metric
            plot_data, labels, colors = [], [], []
            if target in all_results:
                for method in methods_spec.keys():
                    if (method in all_results[target] and 
                        len(all_results[target][method][metric_name]) > 5):
                        plot_data.append(all_results[target][method][metric_name])
                        labels.append(method)
                        colors.append(COLORS.get(method, "gray"))
            
            if not plot_data:
                ax.text(0.5, 0.5, f'{target}\n(Insufficient data)',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Violin plot
            positions = np.arange(1, len(plot_data) + 1)
            parts = ax.violinplot(plot_data, positions=positions,
                                 showmeans=True, showmedians=True, widths=0.6)
            
            for pc, color in zip(parts['bodies'], colors):
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
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel(f"{target}\n(n={n_compounds}, +={n_actives})",
                         fontsize=9, fontweight='bold')
            
            if idx == len(top_targets) - 1:
                ax.set_xlabel("Method", fontsize=10)
            
            style_axes(ax)
            
            if metric_name in ['ROC-AUC', 'BEDROC(20)', 'NEF1%']:
                ax.set_ylim(0, 1.0)
        
        fig.suptitle(f"{metric_name} - Bootstrap Distribution by Target",
                    fontsize=12, fontweight='bold')
        fig.text(0.04, 0.5, metric_name, va='center', rotation='vertical',
                fontsize=11, fontweight='bold')
        
        plt.tight_layout(rect=[0.05, 0, 1, 0.96])
        plt.savefig(os.path.join(outdir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved {filename}")

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(description='Create plots from evaluation results')
    parser.add_argument('--merged', default='results/merged_data.csv', help='Merged data CSV')
    parser.add_argument('--outdir', default='plots', help='Output directory')
    parser.add_argument('--n-boot', type=int, default=200, help='Bootstrap iterations')
    parser.add_argument('--workers', type=int, default=None, help='Parallel workers (default: auto)')
    args = parser.parse_args()
    
    # Load data
    if not os.path.exists(args.merged):
        raise SystemExit(f"Merged data not found: {args.merged}\nRun evaluate_metrics.py first!")
    
    print(f"Loading merged data: {args.merged}")
    merged_df = pd.read_csv(args.merged)
    
    if not {"Protein_ID", "is_active"}.issubset(merged_df.columns):
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
    plot_average_curves(merged_df, methods_spec, args.outdir, max_workers=args.workers)
    
    # Call the new function for per-target plots
    plot_target_curves(merged_df, methods_spec, args.outdir, max_workers=args.workers)
    
    plot_bootstrap_violins(merged_df, methods_spec, args.outdir, 
                          n_boot=args.n_boot, max_workers=args.workers)
    
    print(f"\n{'='*70}")
    print(f"✓ All plots saved to {args.outdir}/")
    print(f"{'='*70}")
    print("  - average_ROC_curves.png")
    print("  - average_PRC_curves.png")
    print("  - per_target_curves/ (directory with one ROC/PRC plot per target)")
    print("  - BEDROC20_violin.png")
    print("  - NEF1pct_violin.png")
    print("  - ROC-AUC_violin.png")
    print("  - EF1pct_violin.png")
    print("\nDone!")

if __name__ == "__main__":
    main()