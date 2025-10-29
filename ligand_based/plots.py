#!/usr/bin/env python3
# plots.py
# Matplotlib-only, one chart per figure, consistent colors per method.
import os, argparse, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

COLORS = {"USR": "C2", "USRCAT": "C3", "Electroshape": "C4"}

def style_axes(ax):
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def parse_value_cell(s):
    if pd.isna(s): return np.nan
    try:
        txt = str(s).strip()
        if "[" in txt:
            val = txt.split("[",1)[0].strip()
        else:
            val = txt
        return float(val)
    except Exception:
        return np.nan

def extract_values_from_per_df(per_df, method, metric):
    col = f"{method} {metric}"
    if col not in per_df.columns: return np.array([])
    return per_df[col].astype(str).map(parse_value_cell).dropna().to_numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", "-L", default=".", help="project root containing output/")
    args = ap.parse_args()
    base = args.location
    outdir = os.path.join(base, "output")
    scores_path = os.path.join(outdir, "scores.csv")
    per_target_path = os.path.join(outdir, "per_target_metrics.csv")
    global_path = os.path.join(outdir, "global_metrics.csv")
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(scores_path): raise SystemExit(f"Missing {scores_path}")
    if not os.path.exists(per_target_path): raise SystemExit(f"Missing {per_target_path}")
    if not os.path.exists(global_path): raise SystemExit(f"Missing {global_path}")

    scores = pd.read_csv(scores_path)
    per_df = pd.read_csv(per_target_path)
    gdf = pd.read_csv(global_path, index_col=0)

    # detect label column
    label_col = None
    for cand in ("label","is_active","active"):
        if cand in scores.columns:
            label_col = cand; break
    if label_col is None:
        raise SystemExit("No label column found in scores.csv")
    labels = scores[label_col].astype(int).to_numpy()

    method_score_cols = [c for c in scores.columns if c.endswith('_score')]
    methods = [c[:-6] for c in method_score_cols]

    # Global ROC + PR per method (separate files)
    # CRITICAL FIX: Distance scores (lower is better) must be inverted to match evaluate.py
    for col in method_score_cols:
        s = scores[col].astype(float).to_numpy()
        # Invert distance scores to match metric calculations in evaluate.py
        # where scores are inverted before passing to ROC/PR calculations
        s = -s
        try:
            fpr, tpr, _ = roc_curve(labels, s)
            roc_auc = auc(fpr, tpr)
        except Exception:
            fpr, tpr, roc_auc = np.array([0,1]), np.array([0,1]), np.nan
        plt.figure(figsize=(6.5,5))
        plt.plot(fpr, tpr, label=f"AUC {roc_auc:.3f}", color="black", lw=2)
        plt.plot([0,1],[0,1],"--", color="gray")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"Global ROC - {col[:-6]}")
        style_axes(plt.gca()); plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"global_ROC_{col[:-6]}.png"), dpi=300)
        plt.close()

    for col in method_score_cols:
        s = scores[col].astype(float).to_numpy()
        # Invert distance scores to match metric calculations in evaluate.py
        s = -s
        try:
            prec, rec, _ = precision_recall_curve(labels, s)
            pr_auc = auc(rec, prec)
        except Exception:
            prec, rec, pr_auc = np.array([0,1]), np.array([0,1]), np.nan
        plt.figure(figsize=(6.5,5))
        plt.plot(rec, prec, label=f"AUPR {pr_auc:.3f}", color="black", lw=2)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Global PRC - {col[:-6]}")
        style_axes(plt.gca()); plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"global_PRC_{col[:-6]}.png"), dpi=300)
        plt.close()

    # Per-method violin plots (one figure per method)
    metrics = ["ROC-AUC","PR-AUC","BEDROC(20)","NEF1%","NEF5%","NEF10%"]
    for method in methods:
        color = COLORS.get(method, "C0")
        for metric in metrics:
            vals = extract_values_from_per_df(per_df, method, metric)
            if vals.size == 0: continue
            plt.figure(figsize=(5,6))
            parts = plt.violinplot([vals], showmeans=True, showmedians=False)
            for b in parts['bodies']:
                b.set_facecolor(color); b.set_edgecolor('black'); b.set_alpha(0.6)
            plt.scatter(np.ones_like(vals)*1 + (np.random.rand(vals.size)-0.5)*0.08, vals, alpha=0.35, s=10, color=color)
            plt.xticks([1],[metric]); plt.ylabel(metric)
            plt.title(f"{method} — Per-target {metric}")
            style_axes(plt.gca()); plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"violin_{method}_{metric.replace('%','pct').replace(' ','_')}.png"), dpi=300)
            plt.close()

    # Save a simple summary table (global_summary already contains formatted cells)
    print("Wrote plots to", os.path.abspath(plots_dir))

if __name__ == "__main__":
    main()