#!/usr/bin/env python3
"""
plot_results.py
Produces plots for global and per‐target metrics:
  - bar chart of global metric values
  - bar chart or dot‐chart of per‐target ROC-AUC sorted
  - scatter plot of per-target EF@1% vs ROC-AUC
  - boxplot of EF@1% distribution
Requires: pandas, matplotlib, seaborn
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(results_dir="results"):
    gl = pd.read_csv(os.path.join(results_dir, "collected_global.csv"))
    pt = pd.read_csv(os.path.join(results_dir, "collected_per_target.csv"))
    return gl, pt

def plot_global(gl, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    # assume gl has metric columns besides 'n_targets'
    metric_cols = [c for c in gl.columns if c != "n_targets"]
    dfm = gl[metric_cols].T.reset_index()
    dfm.columns = ["Metric", "Value"]
    plt.figure(figsize=(8,4))
    sns.barplot(data=dfm, x="Metric", y="Value", palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.title("Global dataset metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_metrics_bar.png"))
    plt.close()

def plot_per_target(pt, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    # ROC-AUC bar (assuming column is named 'roc_auc')
    if "roc_auc" in pt.columns:
        df_sorted = pt.sort_values("roc_auc", ascending=False)
        plt.figure(figsize=(10,6))
        sns.barplot(data=df_sorted, x="target", y="roc_auc", palette="magma")
        plt.xticks(rotation=90)
        plt.ylabel("ROC-AUC")
        plt.title("Per-target ROC-AUC")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "per_target_roc_auc.png"))
        plt.close()
    # Scatter EF@1% vs ROC-AUC
    if "ef_1pct" in pt.columns and "roc_auc" in pt.columns:
        plt.figure(figsize=(6,6))
        sns.scatterplot(data=pt, x="roc_auc", y="ef_1pct", hue="target", legend=False)
        plt.xlabel("ROC-AUC")
        plt.ylabel("EF@1%")
        plt.title("EF@1% vs ROC-AUC per target")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "scatter_ef1_vs_roc.png"))
        plt.close()
    # Boxplot of EF@1%
    if "ef_1pct" in pt.columns:
        plt.figure(figsize=(6,4))
        sns.boxplot(data=pt, y="ef_1pct")
        plt.title("Distribution of EF@1% across targets")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "boxplot_ef1.png"))
        plt.close()

def main():
    results_dir = "results"
    gl, pt = load_data(results_dir)
    plot_global(gl)
    plot_per_target(pt)
    print(f"Plots saved to ./plots")

if __name__ == "__main__":
    main()
