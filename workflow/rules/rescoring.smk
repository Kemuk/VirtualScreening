"""
rescoring.smk

Snakemake rules for AEV-PLIG machine learning-based rescoring.

AEV-PLIG rescoring workflow:
  1. Prepare CSV with docking scores and file paths
  2. Shard CSV for parallel processing
  3. Submit SLURM array job for AEV-PLIG predictions (GPU)
  4. Merge predictions
  5. Update manifest with rescoring results

Rules:
  - prepare_aev_plig_input: Create AEV-PLIG input CSV
  - shard_aev_plig_csv: Split CSV into shards
  - aev_plig_array: Submit SLURM array for GPU predictions
  - run_aev_plig_shard: Run prediction on single shard (local testing)
  - merge_aev_plig_predictions: Combine shard outputs
  - update_manifest_aev_plig: Update manifest with predictions
  - rescore_all: Complete rescoring stage
"""

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

rule prepare_aev_plig_input:
    """
    Generate AEV-PLIG input CSV from manifest.

    Creates CSV with format: unique_id,pK,sdf_file,pdb_file
    Where unique_id is the compound_key from manifest.
    """
    input:
        manifest = MANIFEST_PATH,
        conversion_checkpoint = "data/logs/conversion/conversion_checkpoint.done",

    output:
        csv = "AEV-PLIG/data/lit_pcba.csv",

    log:
        "data/logs/rescoring/prepare_aev_plig_input.log"

    conda:
        "../envs/vscreen.yaml"

    shell:
        """
        python workflow/scripts/prepare_aev_plig_csv.py \
            --manifest {input.manifest} \
            --output {output.csv} \
            2>&1 | tee {log}
        """


rule shard_aev_plig_csv:
    """
    Split AEV-PLIG input CSV into shards for parallel processing.
    """
    input:
        csv = "AEV-PLIG/data/lit_pcba.csv",

    output:
        shards = expand("AEV-PLIG/data/shards/lit_pcba_shard_{shard}.csv", shard=SHARDS),

    log:
        "data/logs/rescoring/shard_aev_plig_csv.log"

    params:
        num_shards = NUM_SHARDS,
        outdir = "AEV-PLIG/data/shards",

    conda:
        "../envs/vscreen.yaml"

    shell:
        """
        python workflow/scripts/shard_csv.py \
            --input {input.csv} \
            --num-shards {params.num_shards} \
            --outdir {params.outdir} \
            --prefix lit_pcba \
            2>&1 | tee {log}
        """


rule aev_plig_array:
    """
    Submit a SLURM array to run AEV-PLIG predictions on all shards.

    Uses GPU-accelerated processing on the htc cluster.
    """
    input:
        shards = expand("AEV-PLIG/data/shards/lit_pcba_shard_{shard}.csv", shard=SHARDS),

    output:
        touch("data/logs/rescoring/aev_plig_array.done")

    log:
        "data/logs/rescoring/aev_plig_array.log"

    params:
        mode = config.get("mode", "production"),
        model = AEV_PLIG_MODEL,

    conda:
        "../envs/vscreen.yaml"

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
      1. Prepare input CSV from manifest
      2. Shard CSV for parallel processing
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
    Only prepare AEV-PLIG input (no predictions).

    Useful for debugging or manual prediction runs.
    """
    input:
        "AEV-PLIG/data/lit_pcba.csv"

    message:
        "AEV-PLIG input CSV prepared!"


rule rescore_shards_only:
    """
    Prepare and shard AEV-PLIG input (no predictions).

    Useful for debugging or manual prediction runs.
    """
    input:
        expand("AEV-PLIG/data/shards/lit_pcba_shard_{shard}.csv", shard=SHARDS)

    message:
        "AEV-PLIG shards prepared!"
