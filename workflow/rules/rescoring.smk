"""
rescoring.smk

Snakemake rules for AEV-PLIG machine learning-based rescoring.

Simplified rules after consolidation:
  - rescore_all: Rescore all via SLURM array jobs (production)

AEV-PLIG rescoring workflow:
  1. Filter manifest to ligands needing rescoring
  2. Submit SLURM array job for GPU predictions
  3. Merge predictions and update manifest

All orchestration handled by unified submit_stage.py.
"""

MODE = config.get('mode', 'production')


# =============================================================================
# Production Rescoring (SLURM Array Jobs)
# =============================================================================

rule rescore_all:
    """
    Rescore all converted ligands with AEV-PLIG via SLURM array jobs.

    This is the recommended production method for rescoring large numbers of ligands.
    Uses the unified submit_stage.py to handle SLURM job submission.

    AEV-PLIG workflow:
      1. Filter manifest to ligands needing rescoring
      2. Create AEV-PLIG input CSVs
      3. Submit GPU array jobs for neural network predictions
      4. Merge predictions and update manifest

    Requires:
      - Docked ligands (docking_checkpoint.done)
      - AEV-PLIG model available
      - GPU resources
    """
    input:
        manifest = MANIFEST_PATH,
        conversion_checkpoint = "data/logs/conversion/conversion_checkpoint.done",

    output:
        checkpoint = touch("data/logs/rescoring/rescoring_checkpoint.done")

    log:
        "data/logs/rescoring/rescore_all.log"

    conda:
        "../envs/vscreen.yaml"

    params:
        mode = MODE,

    shell:
        """
        python -m workflow.slurm.submit_stage \
            --stage aev_infer \
            --mode {params.mode} \
            --manifest {input.manifest} \
            2>&1 | tee {log}
        """
