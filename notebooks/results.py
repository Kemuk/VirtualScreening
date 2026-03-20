import marimo

__generated_with = "0.20.2"
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
    return BEDROC_ALPHA, FRACS, MANIFEST_PATH


@app.cell
def _(MANIFEST_PATH, pl):
    _raw = pl.read_parquet(MANIFEST_PATH)

    df = _raw.filter(pl.col("rescoring_status") == True)
    if df.is_empty():
        df = _raw.filter(pl.col("docking_status") == True)

    R = 0.001987
    T = 298.0

    # Individual ensemble columns — drop any that are entirely null
    _pred_cols = [f"aev_prediction_{i}" for i in range(10)]
    _pred_cols = [c for c in _pred_cols if c in df.columns and df[c].is_not_null().any()]

    df = df.with_columns(
        (-pl.col("vina_score") / (2.303 * R * T)).alias("vina_pK"),
        pl.mean_horizontal(_pred_cols).alias("aev_plig_pK"),
        *[pl.col(c).alias(c) for c in _pred_cols],
    )

    df = df.filter(
        pl.col("aev_plig_pK").is_not_null() &
        pl.col("vina_pK").is_not_null()
    )

    assay_map = {
        "ALDH1": "biochemical", "IDH1": "biochemical", "VDR": "biochemical",
        "FEN1": "biochemical", "KAT2A": "biochemical", "PKM2": "biochemical",
        "GBA": "biochemical", "ADRB2": "cell-based", "ESR1_ago": "cell-based",
        "ESR1_ant": "cell-based", "MAPK1": "cell-based", "MTORC1": "cell-based",
        "OPRK1": "cell-based", "PPARG": "cell-based", "TP53": "cell-based",
    }

    df = df.with_columns(
        pl.col("protein_id").replace(assay_map).alias("assay_type")
    )

    PRED_COLS = _pred_cols
    df, assay_map, PRED_COLS
    return (df,)


@app.cell
def _():
    return


@app.cell
def _(
    BEDROC_ALPHA,
    FRACS,
    HAS_RDKIT,
    RDScoring,
    average_precision_score,
    math,
    np,
    roc_auc_score,
):
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

    return (compute_metrics,)


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
    return (per_target,)


@app.cell
def _(mo):
    mo.md("""
    ### Per-target metrics
    """)
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
    return


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


@app.cell
def _():
    return


@app.cell
def _(df, mo, np, pl):
    from scipy.stats import mannwhitneyu

    def _rank_biserial_r(U, n1, n2):
        return 1 - (2 * U) / (n1 * n2)

    _assay_map = {
        "ALDH1": "biochemical", "IDH1": "biochemical", "VDR": "biochemical",
        "FEN1": "biochemical", "KAT2A": "biochemical", "PKM2": "biochemical",
        "GBA": "biochemical", "ADRB2": "cell-based", "ESR1_ago": "cell-based",
        "ESR1_ant": "cell-based", "MAPK1": "cell-based", "MTORC1": "cell-based",
        "OPRK1": "cell-based", "PPARG": "cell-based", "TP53": "cell-based",
    }

    _rows = []

    for (_target,), _grp in df.partition_by("protein_id", as_dict=True).items():
        _labels = _grp["is_active"].cast(pl.Int8).to_numpy()

        for _method, _col in [("Vina", "vina_pK"), ("AEV-PLIG", "aev_plig_pK")]:
            _scores = _grp[_col].to_numpy()
            _valid  = np.isfinite(_scores)
            _lbl    = _labels[_valid]
            _scr    = _scores[_valid]

            _actives   = _scr[_lbl == 1]
            _inactives = _scr[_lbl == 0]

            if len(_actives) < 2 or len(_inactives) < 2:
                continue

            _U, _p = mannwhitneyu(_actives, _inactives, alternative="greater")
            _r = _rank_biserial_r(_U, len(_actives), len(_inactives))

            _rows.append({
                    "Target":      _target,
                    "assay_type":  _assay_map.get(_target, "unknown"),
                    "Method":      _method,
                    "N actives":   len(_actives),
                    "N inactives": len(_inactives),
                    "U statistic": round(_U, 1),
                    "p-value":     round(_p, 4),
                    "r (effect)":  round(_r, 3),
                    "Significant": "✓" if _p < 0.05 else "✗",
                })

    per_protein_tests = pl.DataFrame(_rows)

    _summary = (
        per_protein_tests
        .group_by(["assay_type", "Method"])
        .agg([
            pl.len().alias("N targets"),
            pl.median("r (effect)").round(3).alias("median r"),
            pl.col("r (effect)").min().round(3).alias("min r"),
            pl.col("r (effect)").max().round(3).alias("max r"),
            (pl.col("p-value") < 0.05).sum().alias("N significant"),
        ])
        .sort(["assay_type", "Method"])
    )

    mo.vstack([
        mo.md("### Per-protein Mann-Whitney U (actives vs inactives, one-sided)"),
        mo.md("Positive r → actives rank higher than inactives; r close to 1 → perfect separation."),
        mo.ui.table(per_protein_tests, selection=None),
        mo.md("### Assay-type summary (median effect size across proteins)"),
        mo.ui.table(_summary, selection=None),
    ])
    return (per_protein_tests,)


@app.cell
def _(mo, per_protein_tests, pl, plt):
    _assay_order = ["biochemical", "cell-based"]
    _method_colours = {"Vina": "#4C72B0", "AEV-PLIG": "#DD8452"}
    _alpha = 0.05

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for _ci, _assay in enumerate(_assay_order):
        _ax = _axes[_ci]
        _grp = (
            per_protein_tests
            .filter(pl.col("assay_type") == _assay)
            .sort("Target")
        )
        _targets = _grp.filter(pl.col("Method") == "Vina")["Target"].to_list()

        for _t in _targets:
            _rows = _grp.filter(pl.col("Target") == _t)
            _r_vina = _rows.filter(pl.col("Method") == "Vina")["r (effect)"][0]
            _r_aev  = _rows.filter(pl.col("Method") == "AEV-PLIG")["r (effect)"][0]
            _ax.plot([0, 1], [_r_vina, _r_aev], color="grey", alpha=0.4, linewidth=1)

        for _method in ["Vina", "AEV-PLIG"]:
            _m = _grp.filter(pl.col("Method") == _method)
            _x = 0 if _method == "Vina" else 1
            _sig = _m["p-value"].to_numpy() < _alpha
            _r   = _m["r (effect)"].to_numpy()
            _t   = _m["Target"].to_list()

            _ax.scatter(
                [_x] * len(_r), _r,
                color=_method_colours[_method],
                marker="o", s=60, zorder=3,
                alpha=0.9, label=_method,
            )
            for _ti, (_ri, _si) in enumerate(zip(_r, _sig)):
                _ax.text(
                    _x + 0.04, _ri, _t[_ti],
                    va="center", fontsize=7, color="grey",
                )

        _ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        _ax.set_xticks([0, 1])
        _ax.set_xticklabels(["Vina", "AEV-PLIG"])
        _ax.set_title(_assay)
        _ax.set_xlim(-0.4, 1.8)
        if _ci == 0:
            _ax.set_ylabel("Rank-biserial r  (actives vs inactives)")

    _axes[0].legend(loc="lower right", fontsize=8)
    _fig.suptitle("Discriminative ability per protein — Mann-Whitney effect size", y=1.02)
    plt.tight_layout()

    mo.vstack([
        mo.md("### Effect sizes by assay type and method"),
        mo.mpl.interactive(_fig),
    ])
    return


if __name__ == "__main__":
    app.run()
