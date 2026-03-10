import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full", app_title="Virtual Screening Results")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from pathlib import Path
    from sklearn.metrics import roc_auc_score, average_precision_score

    try:
        from rdkit.ML.Scoring import Scoring as RDScoring
        HAS_RDKIT = True
    except ImportError:
        HAS_RDKIT = False

    return (
        HAS_RDKIT,
        Path,
        RDScoring,
        average_precision_score,
        gridspec,
        math,
        mo,
        np,
        pl,
        plt,
        roc_auc_score,
    )


@app.cell
def _(Path, mo):
    PROJECT_ROOT = Path(__file__).parent.parent
    MANIFEST_PATH = PROJECT_ROOT / "data" / "master" / "manifest.parquet"
    BEDROC_ALPHA = 20.0
    FRACS = [0.01, 0.05, 0.10]

    mo.md(f"""
    ## Virtual Screening Results

    **Manifest:** `{MANIFEST_PATH}`

    Comparing Vina docking scores (converted to pK binding affinity) against
    AEV-PLIG rescoring predictions across all targets.
    """)

    return BEDROC_ALPHA, FRACS, MANIFEST_PATH, PROJECT_ROOT


@app.cell
def _(MANIFEST_PATH, pl):
    _raw = pl.read_parquet(MANIFEST_PATH)

    # Prefer rescored entries; fall back to docked-only
    df = _raw.filter(pl.col("rescoring_status") == True)
    _source = "rescored"
    if df.is_empty():
        df = _raw.filter(pl.col("docking_status") == True)
        _source = "docked (no rescoring)"

    # Both scores expressed as pK (higher = better binding)
    # binding_affinity_pK = -vina_score / (2.303 * R * T) is pre-computed in the manifest
    # aev_plig_best_score is already in pK units
    df = df.with_columns(
        pl.col("binding_affinity_pK").alias("vina_pK"),
        pl.col("aev_plig_best_score").alias("aev_plig_pK"),
    )

    _source, df

    return df,


@app.cell
def _(df, mo):
    _s = df.select(
        pl.col("protein_id").n_unique().alias("Targets"),
        pl.len().alias("Compounds"),
        pl.col("is_active").sum().cast(pl.Int64).alias("Actives"),
        (pl.col("is_active").mean() * 100).round(1).alias("Active %"),
        pl.col("vina_pK").is_not_null().sum().alias("Vina scores"),
        pl.col("aev_plig_pK").is_not_null().sum().alias("AEV-PLIG scores"),
    ).row(0, named=True)

    mo.hstack([
        mo.stat(value=str(_s["Targets"]),      label="Targets"),
        mo.stat(value=f'{_s["Compounds"]:,}',  label="Compounds"),
        mo.stat(value=f'{_s["Actives"]:,}',    label="Actives"),
        mo.stat(value=f'{_s["Active %"]}%',    label="Active rate"),
        mo.stat(value=f'{_s["Vina scores"]:,}',     label="Vina scores"),
        mo.stat(value=f'{_s["AEV-PLIG scores"]:,}', label="AEV-PLIG scores"),
    ])

    return


@app.cell
def _(BEDROC_ALPHA, FRACS, HAS_RDKIT, RDScoring, average_precision_score,
       math, np, roc_auc_score):
    def _rdkit_table(labels, scores):
        """Sort descending (higher-is-better) for RDKit metrics."""
        order = np.argsort(scores)[::-1]
        return list(map(list, zip(scores[order].astype(float),
                                   labels[order].astype(bool))))

    def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
        """
        Compute ROC-AUC, PR-AUC, BEDROC, EF1/5/10% for one target-method pair.
        Both labels and scores must be clean (no NaN); higher score = better.
        """
        if len(np.unique(labels)) < 2 or len(labels) < 10:
            return {}

        metrics = {
            "ROC-AUC": float(roc_auc_score(labels, scores)),
            "PR-AUC":  float(average_precision_score(labels, scores)),
        }

        if HAS_RDKIT:
            arr = _rdkit_table(labels, scores)
            metrics["BEDROC"] = float(RDScoring.CalcBEDROC(arr, 1, BEDROC_ALPHA))
            for frac in FRACS:
                ef = float(RDScoring.CalcEnrichment(arr, 1, [frac])[0])
                metrics[f"EF{int(frac * 100)}%"] = ef
                # NEF: normalise by theoretical maximum
                n_pos  = int(labels.sum())
                top_n  = max(1, math.ceil(len(labels) * frac))
                ef_max = (min(n_pos, top_n) / top_n) / (n_pos / len(labels))
                metrics[f"NEF{int(frac * 100)}%"] = float(
                    np.clip(ef / ef_max, 0, 1) if ef_max > 0 else 0
                )

        return metrics

    return compute_metrics,


@app.cell
def _(compute_metrics, df, np, pl):
    _rows = []

    for (_target,), _group in df.partition_by("protein_id", as_dict=True).items():
        _labels = _group["is_active"].cast(pl.Int8).to_numpy().astype(int)
        _n      = len(_labels)
        _n_act  = int(_labels.sum())

        for _method, _col in [("Vina", "vina_pK"), ("AEV-PLIG", "aev_plig_pK")]:
            _raw_scores = _group[_col].to_numpy()
            _valid      = np.isfinite(_raw_scores) & np.isfinite(_labels.astype(float))
            _lbl        = _labels[_valid]
            _scr        = _raw_scores[_valid]

            _m = compute_metrics(_lbl, _scr)
            if _m:
                _rows.append({
                    "Target":      _target,
                    "Method":      _method,
                    "N":           _n,
                    "N_Actives":   _n_act,
                    **{k: round(v, 4) for k, v in _m.items()},
                })

    per_target = pl.DataFrame(_rows)
    per_target

    return per_target,


@app.cell
def _(mo, per_target):
    mo.md("### Per-target metrics")

    return


@app.cell
def _(mo, per_target):
    mo.ui.table(per_target, selection=None)

    return


@app.cell
def _(mo, per_target, pl):
    # Aggregate across targets: median per method for each metric
    _metric_cols = [c for c in per_target.columns
                    if c not in ("Target", "Method", "N", "N_Actives")]

    _agg = (
        per_target
        .group_by("Method")
        .agg([pl.median(c).round(4).alias(c) for c in _metric_cols])
        .sort("Method")
    )

    mo.vstack([
        mo.md("### Aggregated metrics (median across targets)"),
        mo.ui.table(_agg, selection=None),
    ])

    return


@app.cell
def _(df, mo, np, per_target, pl, plt):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── 1. ROC-AUC comparison per target ────────────────────────────────────
    _vina_roc = (
        per_target.filter(pl.col("Method") == "Vina")
        .sort("Target")["ROC-AUC"]
        .to_numpy()
    )
    _aev_roc = (
        per_target.filter(pl.col("Method") == "AEV-PLIG")
        .sort("Target")["ROC-AUC"]
        .to_numpy()
    )
    _targets = (
        per_target.filter(pl.col("Method") == "Vina")
        .sort("Target")["Target"]
        .to_list()
    )
    _x = np.arange(len(_targets))
    _w = 0.35
    axes[0].bar(_x - _w / 2, _vina_roc,  _w, label="Vina",     color="#4C72B0")
    axes[0].bar(_x + _w / 2, _aev_roc,   _w, label="AEV-PLIG", color="#DD8452")
    axes[0].set_xticks(_x)
    axes[0].set_xticklabels(_targets, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("ROC-AUC per target")
    axes[0].axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    # ── 2. Score distributions for actives vs inactives ─────────────────────
    _actives   = df.filter(pl.col("is_active") == True)
    _inactives = df.filter(pl.col("is_active") == False)

    for _col, _label, _color in [
        ("vina_pK",    "Vina pK",     "#4C72B0"),
        ("aev_plig_pK","AEV-PLIG pK", "#DD8452"),
    ]:
        _act_scores = _actives[_col].drop_nulls().to_numpy()
        _ina_scores = _inactives[_col].drop_nulls().to_numpy()
        _ax = axes[1] if _col == "vina_pK" else axes[2]
        _ax.hist(_ina_scores, bins=60, density=True, alpha=0.5,
                 color="grey",    label="Inactives")
        _ax.hist(_act_scores,  bins=60, density=True, alpha=0.7,
                 color=_color,   label="Actives")
        _ax.set_xlabel(f"{_label}")
        _ax.set_ylabel("Density")
        _ax.set_title(f"{_label}: actives vs inactives")
        _ax.legend(fontsize=8)

    plt.tight_layout()
    mo.mpl.interactive(fig)

    return axes, fig


@app.cell
def _(df, mo, pl, plt):
    # Scatter: vina_pK vs aev_plig_pK, coloured by activity
    _plot_df = df.select(["vina_pK", "aev_plig_pK", "is_active"]).drop_nulls()
    _act = _plot_df.filter(pl.col("is_active") == True)
    _ina = _plot_df.filter(pl.col("is_active") == False)

    _fig2, _ax2 = plt.subplots(figsize=(6, 5))
    _ax2.scatter(_ina["vina_pK"].to_numpy(), _ina["aev_plig_pK"].to_numpy(),
                 alpha=0.2, s=4, color="grey",    label="Inactives")
    _ax2.scatter(_act["vina_pK"].to_numpy(), _act["aev_plig_pK"].to_numpy(),
                 alpha=0.6, s=8, color="#DD8452", label="Actives")
    _ax2.set_xlabel("Vina pK (binding affinity)")
    _ax2.set_ylabel("AEV-PLIG pK")
    _ax2.set_title("Vina pK vs AEV-PLIG pK")
    _ax2.legend()
    plt.tight_layout()
    mo.mpl.interactive(_fig2)

    return


if __name__ == "__main__":
    app.run()
