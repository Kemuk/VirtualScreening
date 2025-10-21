#!/usr/bin/env python3
"""
results_collection.py
Loads:
  - results/dataset_summary.csv (global metrics for whole dataset)
  - results/per_target_metrics.csv (one row per target)
Writes:
  - results/collected_per_target.csv (cleaned per-target)
  - results/collected_global.csv (cleaned global metrics)
"""
import os
import pandas as pd

def load_files(results_dir="results"):
    ds_path = os.path.join(results_dir, "dataset_summary.csv")
    pt_path = os.path.join(results_dir, "per_target_metrics.csv")

    if not os.path.isfile(ds_path):
        raise FileNotFoundError(f"Global summary file not found at {ds_path}")
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"Per-target metrics file not found at {pt_path}")

    df_global = pd.read_csv(ds_path)
    df_pt = pd.read_csv(pt_path)
    return df_global, df_pt

def clean_per_target(df_pt):
    # ensure target column exists
    if "target" not in df_pt.columns:
        raise KeyError("Expected 'target' column in per‐target metrics CSV")
    # select numeric metric columns (assuming they are floats or NaNs)
    metric_cols = [c for c in df_pt.columns if c not in ["target"]]
    df_clean = df_pt[["target"] + metric_cols].copy()
    return df_clean

def clean_global(df_global):
    return df_global.copy()

def main():
    results_dir = "results"
    df_global, df_pt = load_files(results_dir)
    df_pt_clean = clean_per_target(df_pt)
    df_global_clean = clean_global(df_global)

    # write cleaned files
    out_pt = os.path.join(results_dir, "collected_per_target.csv")
    out_gl = os.path.join(results_dir, "collected_global.csv")
    df_pt_clean.to_csv(out_pt, index=False)
    df_global_clean.to_csv(out_gl, index=False)

    print(f"Wrote cleaned per‐target metrics to {out_pt}")
    print(f"Wrote cleaned global metrics to {out_gl}")

if __name__ == "__main__":
    main()
