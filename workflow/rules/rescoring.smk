"""
rescoring.smk

Snakemake rules for AEV-PLIG machine learning-based rescoring.

AEV-PLIG rescoring workflow:
  1. Shard manifest entries needing rescoring
  2. Build AEV-PLIG shard CSVs from rescoring chunks
  3. Submit SLURM array job for AEV-PLIG predictions (GPU)
  4. Merge predictions
  5. Update manifest with rescoring results

Rules:
  - shard_rescoring: Create rescoring chunk CSVs from manifest
  - prepare_aev_plig_shard: Build AEV-PLIG shard CSVs from rescoring chunks
  - aev_plig_array: Submit SLURM array for GPU predictions
  - run_aev_plig_shard: Run prediction on single shard (local testing)
  - merge_aev_plig_predictions: Combine shard outputs
  - update_manifest_aev_plig: Update manifest with predictions
  - rescore_all: Complete rescoring stage
"""

import os
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

# Get mode-specific settings
MODE = config.get('mode', 'production')
MODE_CONFIG = config.get(MODE, {})

# Number of shards for parallel AEV-PLIG processing
chunking = config.get('chunking', {})
mode_chunking = chunking.get(MODE, {})
if MODE == "production":
    NUM_SHARDS = mode_chunking.get('gpu_chunks', MODE_CONFIG.get('aev_plig_shards', 100))
else:
    NUM_SHARDS = mode_chunking.get('chunks', MODE_CONFIG.get('aev_plig_shards', 5))
SHARDS = list(range(NUM_SHARDS))

# AEV-PLIG settings
AEV_PLIG_DIR = config.get('tools', {}).get('aev_plig_dir', 'AEV-PLIG')
AEV_PLIG_MODEL = config.get('rescoring', {}).get('model_name', 'model_GATv2Net_ligsim90_fep_benchmark')
AEV_PLIG_CONDA = os.environ.get(
    "AEV_PLIG_CONDA",
    os.path.join(os.environ["DATA"], "aev-plig") if os.environ.get("DATA") else "aev-plig",
)


# =============================================================================
# Helper Functions
# =============================================================================

def get_targets_needing_rescoring():
    """
    Get list of targets that have docked ligands ready for rescoring.

    Returns:
        List of target IDs
    """
    manifest = load_manifest()

    # Find targets with at least one docked ligand
    docked = manifest[manifest['docking_status'] == True]
    targets_with_docking = docked['protein_id'].unique().tolist()

    return targets_with_docking


# =============================================================================
# AEV-PLIG Full Pipeline with Sharding
# =============================================================================

rule shard_rescoring:
    """
    Create rescoring chunk CSVs from the manifest (only missing AEV-PLIG scores).
    """
    input:
        manifest = MANIFEST_PATH,

    output:
        expand("data/chunks/rescoring/chunk_{chunk}.csv", chunk=SHARDS),

    log:
        "data/logs/rescoring/shard_rescoring.log"

    params:
        num_chunks = NUM_SHARDS,

    conda:
        "../envs/vscreen.yaml"

    shell:
        """
        python workflow/scripts/shard_stage.py \
            --stage rescoring \
            --manifest {input.manifest} \
            --outdir data/chunks/rescoring \
            --num-chunks {params.num_chunks} \
            2>&1 | tee {log}
        """


rule prepare_aev_plig_shard:
    """
    Build AEV-PLIG shard CSVs from rescoring chunks.

    NOTE: Kept for local/manual testing. For cluster execution, use
    prepare_aev_plig_array to submit an array job.
    """
    input:
        chunk = "data/chunks/rescoring/chunk_{shard}.csv",

    output:
        shard = "AEV-PLIG/data/shards/lit_pcba_shard_{shard}.csv",

    log:
        "data/logs/rescoring/prepare_aev_plig_shard_{shard}.log"

    conda:
        "../envs/vscreen.yaml"

    shell:
        """
        python workflow/scripts/prepare_aev_plig_shard.py \
            --chunk {input.chunk} \
            --output {output.shard} \
            2>&1 | tee {log}
        """


rule prepare_aev_plig_array:
    """
    Submit a SLURM array to build AEV-PLIG shard CSVs from rescoring chunks.
    """
    input:
        expand("data/chunks/rescoring/chunk_{chunk}.csv", chunk=SHARDS),

    output:
        touch("data/logs/rescoring/prepare_aev_plig_array.done")

    log:
        "data/logs/rescoring/prepare_aev_plig_array.log"

    params:
        mode = config.get("mode", "production"),

    conda:
        "../envs/vscreen.yaml"

    shell:
        """
        bash workflow/scripts/submit_prepare_aev_plig_array.sh \
            --chunks-dir data/chunks/rescoring \
            --output-dir AEV-PLIG/data/shards \
            --log-dir data/logs/rescoring \
            --slurm-log-dir data/logs/slurm \
            --config config/config.yaml \
            --mode {params.mode} \
            2>&1 | tee {log}
        """


rule aev_plig_array:
    """
    Submit a SLURM array to run AEV-PLIG predictions on all shards.

    Uses GPU-accelerated processing on the htc cluster.
    """
    input:
        shards_done = "data/logs/rescoring/prepare_aev_plig_array.done",

    output:
        touch("data/logs/rescoring/aev_plig_array.done")

    log:
        "data/logs/rescoring/aev_plig_array.log"

    params:
        mode = config.get("mode", "production"),
        model = AEV_PLIG_MODEL,

    conda:
        AEV_PLIG_CONDA

    shell:
        """
        mkdir -p AEV-PLIG/output/shards
        bash workflow/scripts/submit_aev_plig_array.sh \
            --shards-dir AEV-PLIG/data/shards \
            --output-dir AEV-PLIG/output/shards \
            --log-dir data/logs/rescoring \
            --slurm-log-dir data/logs/slurm \
            --config config/config.yaml \
            --mode {params.mode} \
            --model {params.model} \
            2>&1 | tee {log}
        """


rule run_aev_plig_shard:
    """
    Run AEV-PLIG neural network predictions on a single shard.

    NOTE: This rule is kept for local/manual testing. For cluster execution,
    use aev_plig_array which submits a SLURM array job.

    Executes the AEV-PLIG model on one shard of data.
    Output contains predictions from 10 model ensembles plus final prediction.
    """
    input:
        csv = "AEV-PLIG/data/shards/lit_pcba_shard_{shard}.csv",

    output:
        predictions = "AEV-PLIG/output/shards/shard_{shard}_predictions.csv",

    log:
        "data/logs/rescoring/run_aev_plig_shard_{shard}.log"

    params:
        aev_plig_dir = AEV_PLIG_DIR,
        model = AEV_PLIG_MODEL,
        shard = lambda wildcards: wildcards.shard,

    conda:
        AEV_PLIG_CONDA

    shell:
        """
        mkdir -p {params.aev_plig_dir}/output/shards
        cd {params.aev_plig_dir} && \
        python process_and_predict.py \
            --dataset_csv=data/shards/lit_pcba_shard_{params.shard}.csv \
            --data_name=shard_{params.shard} \
            --trained_model_name={params.model} \
            2>&1 | tee ../{log}

        # Move output to expected location
        mv output/predictions/lit_pcba_shard_{params.shard}_predictions.csv \
           output/shards/shard_{params.shard}_predictions.csv 2>/dev/null || \
        mv output/predictions/shard_{params.shard}_predictions.csv \
           output/shards/shard_{params.shard}_predictions.csv 2>/dev/null || true
        """


rule merge_aev_plig_predictions:
    """
    Merge all shard predictions into a single CSV file.
    """
    input:
        array_done = "data/logs/rescoring/aev_plig_array.done",

    output:
        merged = "AEV-PLIG/output/predictions/lit_pcba_predictions.csv",

    log:
        "data/logs/rescoring/merge_aev_plig_predictions.log"

    params:
        shards_dir = "AEV-PLIG/output/shards",
        num_shards = NUM_SHARDS,

    run:
        import pandas as pd
        from pathlib import Path

        shards_dir = Path(params.shards_dir)
        shard_files = [shards_dir / f"shard_{i}_predictions.csv" for i in range(params.num_shards)]

        print(f"Merging {len(shard_files)} shard predictions...")

        dfs = []
        for shard_file in shard_files:
            try:
                df = pd.read_csv(shard_file)
                dfs.append(df)
                print(f"  Loaded {shard_file}: {len(df)} rows")
            except Exception as e:
                print(f"  WARNING: Failed to load {shard_file}: {e}")

        if not dfs:
            raise ValueError("No shard files could be loaded!")

        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.to_csv(output.merged, index=False)

        print(f"\nMerged {len(merged_df)} total predictions to {output.merged}")


rule update_manifest_aev_plig:
    """
    Update manifest with AEV-PLIG predictions.

    Adds to manifest:
      - binding_affinity_pK: converted from Vina score
      - aev_plig_best_score: ensemble prediction
      - aev_prediction_0-9: individual model predictions
      - rescoring_status: set to True
    """
    input:
        manifest = MANIFEST_PATH,
        predictions = "AEV-PLIG/output/predictions/lit_pcba_predictions.csv",

    output:
        done = touch("data/logs/rescoring/rescoring_checkpoint.done"),

    log:
        "data/logs/rescoring/update_manifest_aev_plig.log"

    conda:
        "../envs/vscreen.yaml"

    shell:
        """
        python workflow/scripts/update_manifest_aev_plig.py \
            --manifest {input.manifest} \
            --predictions {input.predictions} \
            2>&1 | tee {log}
        """


# =============================================================================
# Non-Sharded Pipeline (for small datasets or local testing)
# =============================================================================

rule run_aev_plig_single:
    """
    Run AEV-PLIG neural network predictions without sharding.

    Use this for small datasets or local testing.
    Usage: snakemake run_aev_plig_single
    """
    input:
        csv = "AEV-PLIG/data/lit_pcba.csv",

    output:
        predictions = "AEV-PLIG/output/predictions/lit_pcba_single_predictions.csv",

    log:
        "data/logs/rescoring/run_aev_plig_single.log"

    params:
        aev_plig_dir = AEV_PLIG_DIR,
        model = AEV_PLIG_MODEL,

    conda:
        AEV_PLIG_CONDA

    shell:
        """
        cd {params.aev_plig_dir} && \
        python process_and_predict.py \
            --dataset_csv=data/lit_pcba.csv \
            --data_name=lit_pcba \
            --trained_model_name={params.model} \
            2>&1 | tee ../{log}
        """


# =============================================================================
# Convenience Rules
# =============================================================================

rule rescore_all:
    """
    Complete rescoring stage with sharded AEV-PLIG predictions.

    This runs the full AEV-PLIG pipeline:
      1. Shard manifest rows needing rescoring
      2. Prepare AEV-PLIG shard CSVs
      3. Run AEV-PLIG neural network on each shard
      4. Merge predictions
      5. Update manifest with predictions
    """
    input:
        "data/logs/rescoring/rescoring_checkpoint.done"

    message:
        "AEV-PLIG rescoring complete!"


rule rescore_prepare_only:
    """
    Only create rescoring chunk CSVs (no predictions).

    Useful for debugging which ligands are selected for rescoring.
    """
    input:
        expand("data/chunks/rescoring/chunk_{chunk}.csv", chunk=SHARDS)

    message:
        "Rescoring chunks prepared!"


rule rescore_shards_only:
    """
    Prepare AEV-PLIG shard CSVs (no predictions).

    Useful for debugging or manual prediction runs.
    """
    input:
        "data/logs/rescoring/prepare_aev_plig_array.done"

    message:
        "AEV-PLIG shards prepared!"
