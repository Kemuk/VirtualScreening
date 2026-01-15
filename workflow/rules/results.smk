"""
results.smk

Snakemake rules for computing and visualizing virtual screening results.

Results workflow:
  1. Compute per-target metrics (ROC-AUC, BEDROC, EF, NEF)
  2. Aggregate with bootstrap confidence intervals
  3. Create visualization plots

Rules:
  - compute_results: Calculate metrics from manifest
  - make_plots: Generate visualization plots
  - results_all: Complete results stage
"""


# =============================================================================
# Configuration
# =============================================================================

RESULTS_DIR = config.get('results_dir', 'results')
RESULTS_CONFIG = config.get('results', {})

BOOTSTRAP_ITERATIONS = RESULTS_CONFIG.get('bootstrap_iterations', 1000)
BEDROC_ALPHA = RESULTS_CONFIG.get('bedroc_alpha', 20.0)
ENRICHMENT_FRACTIONS = RESULTS_CONFIG.get('enrichment_fractions', [0.01, 0.05, 0.10])


# =============================================================================
# Results Computation
# =============================================================================

rule compute_results:
    """
    Compute virtual screening metrics from manifest.

    Calculates per-target metrics and aggregates with bootstrap CIs.
    Outputs:
      - per_target_metrics.csv: Metrics for each target
      - summary.csv: Aggregated metrics with CIs and p-values
    """
    input:
        manifest = MANIFEST_PATH,
        rescoring_done = "data/logs/rescoring/aev_plig_complete.done",

    output:
        per_target = f"{RESULTS_DIR}/per_target_metrics.csv",
        summary = f"{RESULTS_DIR}/summary.csv",

    log:
        "data/logs/results/compute_results.log"

    params:
        outdir = RESULTS_DIR,
        fracs = ",".join([str(f) for f in ENRICHMENT_FRACTIONS]),
        bedroc_alpha = BEDROC_ALPHA,
        n_boot = BOOTSTRAP_ITERATIONS,

    conda:
        "../envs/vscreen.yaml"

    shell:
        """
        python workflow/scripts/compute_results.py \
            --manifest {input.manifest} \
            --outdir {params.outdir} \
            --fracs {params.fracs} \
            --bedroc-alpha {params.bedroc_alpha} \
            --n-boot {params.n_boot} \
            2>&1 | tee {log}
        """


rule make_plots:
    """
    Create visualization plots from results.

    Generates:
      - Average ROC and PRC curves
      - Per-target ROC/PRC curves
      - Metric comparison plots
      - Violin distribution plots
    """
    input:
        manifest = MANIFEST_PATH,
        per_target = f"{RESULTS_DIR}/per_target_metrics.csv",

    output:
        directory(f"{RESULTS_DIR}/plots"),

    log:
        "data/logs/results/make_plots.log"

    params:
        outdir = f"{RESULTS_DIR}/plots",

    conda:
        "../envs/vscreen.yaml"

    shell:
        """
        mkdir -p {params.outdir}
        python workflow/scripts/make_results_plots.py \
            --manifest {input.manifest} \
            --metrics {input.per_target} \
            --outdir {params.outdir} \
            2>&1 | tee {log}
        """


# =============================================================================
# Convenience Rules
# =============================================================================

rule results_all:
    """
    Complete results stage: compute metrics and create plots.
    """
    input:
        f"{RESULTS_DIR}/per_target_metrics.csv",
        f"{RESULTS_DIR}/summary.csv",
        f"{RESULTS_DIR}/plots",

    message:
        "Results computation and visualization complete!"


rule results_metrics_only:
    """
    Only compute metrics (no plots).
    """
    input:
        f"{RESULTS_DIR}/per_target_metrics.csv",
        f"{RESULTS_DIR}/summary.csv",

    message:
        "Metrics computation complete!"


rule results_plots_only:
    """
    Only create plots (assumes metrics already computed).
    """
    input:
        f"{RESULTS_DIR}/plots",

    message:
        "Plot generation complete!"
