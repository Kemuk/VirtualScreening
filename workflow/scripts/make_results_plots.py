#!/usr/bin/env python3
"""
make_results_plots.py

Create publication-ready plots from virtual screening results.

Reads manifest parquet and creates:
  - Average ROC curves
  - Average Precision-Recall curves
  - Per-target ROC/PRC curves
  - Bootstrap violin plots for key metrics

Outputs saved to results/plots/ directory.
"""

import argparse
import sys
import os
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve, auc

try:
    from rdkit.ML.Scoring import Scoring as RDScoring
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# Plot settings
plt.rcParams.update({
    "font.size": 11,
    "figure.figsize": (8, 6),
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "Vina": "#3498db",
    "AEV-PLIG": "#e74c3c"
}


def style_axes(ax):
    """Apply consistent styling to axes."""
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# =============================================================================
# Curve Computation
# =============================================================================

def compute_curves_worker(args):
    """Compute ROC and PRC curves for one target-method combination."""
    target, method, labels, scores, higher_is_better = args

    labels = np.array(labels)
    scores = np.array(scores)

    # Remove NaN
    valid = ~np.isnan(scores)
    labels, scores = labels[valid], scores[valid]

    if len(labels) < 10 or len(np.unique(labels)) < 2:
        return None

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
            'baseline': baseline
        }
    except Exception:
        return None


# =============================================================================
# Plot Functions
# =============================================================================

def plot_average_curves(df, methods_spec, outdir, max_workers=None):
    """Create average ROC and PRC curves."""
    print("\nCreating average ROC and PRC curves...")

    # Prepare tasks
    tasks = []
    for method, spec in methods_spec.items():
        if method not in df.columns:
            continue

        for target, group in df.groupby("protein_id", sort=False):
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
            std_auc = np.std(roc_data[method]['aucs'])
            plt.plot(common_fpr, mean_tpr, color=COLORS.get(method, 'gray'), lw=2,
                     label=f"{method} (AUC={mean_auc:.3f}±{std_auc:.3f})")

    plt.plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Average ROC Curves Across Targets")
    plt.legend(loc="lower right")
    style_axes(plt.gca())
    plt.tight_layout()
    plt.savefig(outdir / "average_ROC.png")
    plt.close()
    print("  Saved average_ROC.png")

    # Plot PRC
    plt.figure(figsize=(8, 6))
    common_recall = np.linspace(0, 1, 201)
    for method in methods_spec.keys():
        if prc_data[method]['precs']:
            mean_prec = np.mean(prc_data[method]['precs'], axis=0)
            mean_auc = np.mean(prc_data[method]['aucs'])
            std_auc = np.std(prc_data[method]['aucs'])
            plt.plot(common_recall, mean_prec, color=COLORS.get(method, 'gray'), lw=2,
                     label=f"{method} (AUPR={mean_auc:.3f}±{std_auc:.3f})")

    baseline = df["is_active"].mean()
    plt.axhline(baseline, color='gray', linestyle='--', linewidth=1,
                label=f'Random ({baseline:.3f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Average Precision-Recall Curves Across Targets")
    plt.legend(loc="upper right")
    style_axes(plt.gca())
    plt.tight_layout()
    plt.savefig(outdir / "average_PRC.png")
    plt.close()
    print("  Saved average_PRC.png")


def plot_per_target_curves(df, methods_spec, outdir, max_workers=None):
    """Create individual ROC and PRC plots for each target."""
    print("\nCreating per-target curves...")

    target_dir = outdir / "per_target_curves"
    target_dir.mkdir(exist_ok=True)

    # Prepare tasks
    tasks = []
    for method, spec in methods_spec.items():
        if method not in df.columns:
            continue

        for target, group in df.groupby("protein_id", sort=False):
            clean = group.dropna(subset=[method, "is_active"])
            if len(clean) < 10 or clean["is_active"].nunique() < 2:
                continue

            labels = clean["is_active"].astype(int).values
            scores = clean[method].astype(float).values
            tasks.append((target, method, labels, scores, spec["higher_is_better"]))

    # Compute curves
    results_by_target = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_curves_worker, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures),
                          desc="  Computing curves", leave=False):
            result = future.result()
            if result:
                target = result['target']
                if target not in results_by_target:
                    results_by_target[target] = []
                results_by_target[target].append(result)

    # Plot each target
    for target, results_list in tqdm(results_by_target.items(),
                                     desc="  Saving plots", leave=False):
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        fig_prc, ax_prc = plt.subplots(figsize=(7, 6))
        target_baseline = None

        for result in results_list:
            method = result['method']
            fpr, tpr, roc_auc = result['roc']
            prec, rec, pr_auc = result['prc']
            target_baseline = result['baseline']

            ax_roc.plot(fpr, tpr, color=COLORS.get(method, 'gray'), lw=2,
                        label=f"{method} (AUC={roc_auc:.3f})")

            sort_idx = np.argsort(rec)
            ax_prc.plot(rec[sort_idx], prec[sort_idx], color=COLORS.get(method, 'gray'), lw=2,
                        label=f"{method} (AUPR={pr_auc:.3f})")

        # Finalize ROC
        ax_roc.plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Random')
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"ROC Curve - {target}")
        ax_roc.legend(loc="lower right")
        style_axes(ax_roc)
        fig_roc.tight_layout()
        fig_roc.savefig(target_dir / f"{target}_ROC.png", dpi=150)
        plt.close(fig_roc)

        # Finalize PRC
        if target_baseline is not None:
            ax_prc.axhline(target_baseline, color='gray', linestyle='--', linewidth=1,
                           label=f'Random ({target_baseline:.3f})')
        ax_prc.set_xlabel("Recall")
        ax_prc.set_ylabel("Precision")
        ax_prc.set_title(f"Precision-Recall Curve - {target}")
        ax_prc.legend(loc="upper right")
        style_axes(ax_prc)
        fig_prc.tight_layout()
        fig_prc.savefig(target_dir / f"{target}_PRC.png", dpi=150)
        plt.close(fig_prc)

    print(f"  Saved {len(results_by_target)} per-target plots")


def plot_metric_comparison(per_target_df, outdir):
    """Create bar chart comparing metrics."""
    print("\nCreating metric comparison plots...")

    methods = ['Vina', 'AEV-PLIG']
    metrics = ['ROC-AUC', 'BEDROC', 'EF1%', 'NEF1%']

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(len(methods))
        width = 0.6

        means = []
        stds = []
        for method in methods:
            col = f'{method}_{metric}'
            if col in per_target_df.columns:
                values = per_target_df[col].dropna()
                means.append(values.mean())
                stds.append(values.std())
            else:
                means.append(0)
                stds.append(0)

        bars = ax.bar(x, means, width, yerr=stds, capsize=5,
                      color=[COLORS.get(m, 'gray') for m in methods],
                      edgecolor='black', linewidth=1.2)

        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison (Mean ± Std across targets)')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        style_axes(ax)

        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(outdir / f"{metric.replace('%', 'pct')}_comparison.png")
        plt.close()

    print("  Saved metric comparison plots")


def plot_violin_distributions(per_target_df, outdir):
    """Create violin plots for metric distributions."""
    print("\nCreating violin plots...")

    methods = ['Vina', 'AEV-PLIG']
    metrics = ['ROC-AUC', 'BEDROC', 'EF1%', 'NEF1%']

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 6))

        plot_data = []
        labels = []
        colors = []

        for method in methods:
            col = f'{method}_{metric}'
            if col in per_target_df.columns:
                values = per_target_df[col].dropna().values
                if len(values) > 0:
                    plot_data.append(values)
                    labels.append(method)
                    colors.append(COLORS.get(method, 'gray'))

        if not plot_data:
            plt.close()
            continue

        positions = np.arange(1, len(plot_data) + 1)
        parts = ax.violinplot(plot_data, positions=positions,
                              showmeans=True, showmedians=True, widths=0.7)

        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        if 'cmeans' in parts:
            parts['cmeans'].set_edgecolor('black')
            parts['cmeans'].set_linewidth(1.5)
        if 'cmedians' in parts:
            parts['cmedians'].set_edgecolor('darkred')
            parts['cmedians'].set_linewidth(1.5)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Distribution Across Targets')
        style_axes(ax)

        plt.tight_layout()
        plt.savefig(outdir / f"{metric.replace('%', 'pct')}_violin.png")
        plt.close()

    print("  Saved violin plots")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create plots from virtual screening results'
    )
    parser.add_argument(
        '--manifest',
        type=Path,
        required=True,
        help='Path to manifest parquet file'
    )
    parser.add_argument(
        '--metrics',
        type=Path,
        default=None,
        help='Path to per_target_metrics.csv (optional)'
    )
    parser.add_argument(
        '--outdir',
        type=Path,
        default=Path('results/plots'),
        help='Output directory'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Parallel workers'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    # Load manifest
    print(f"Loading manifest from {args.manifest}...")
    manifest = pq.read_table(args.manifest).to_pandas()

    # Filter to rescored
    df = manifest[manifest['rescoring_status'] == True].copy()
    if len(df) == 0:
        df = manifest[manifest['docking_status'] == True].copy()

    df['Vina'] = df['vina_score']
    df['AEV-PLIG'] = df['aev_plig_best_score']

    print(f"  Compounds: {len(df)}")
    print(f"  Targets: {df['protein_id'].nunique()}")

    # Load per-target metrics if provided
    per_target_df = None
    if args.metrics and args.metrics.exists():
        per_target_df = pd.read_csv(args.metrics)
        print(f"  Loaded per-target metrics: {len(per_target_df)} targets")

    # Define methods
    methods_spec = {
        "Vina": {"higher_is_better": False},
        "AEV-PLIG": {"higher_is_better": True}
    }

    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Create plots
    plot_average_curves(df, methods_spec, args.outdir, max_workers=args.workers)
    plot_per_target_curves(df, methods_spec, args.outdir, max_workers=args.workers)

    if per_target_df is not None:
        plot_metric_comparison(per_target_df, args.outdir)
        plot_violin_distributions(per_target_df, args.outdir)

    print(f"\nAll plots saved to {args.outdir}/")
    print("Done!")
    sys.exit(0)


if __name__ == "__main__":
    main()
