"""
rescoring.smk

Snakemake rules for AEV-PLIG machine learning-based rescoring.

AEV-PLIG rescoring workflow:
  1. Prepare CSV with molecular properties and docking scores
  2. Run AEV-PLIG neural network rescoring (optional)
  3. Update manifest with rescoring results

Rules:
  - prepare_aev_plig_data: Create AEV-PLIG input CSV for a target
  - prepare_all_aev_plig: Prepare CSV files for all targets
  - rescore_all: Complete rescoring stage
"""

import pandas as pd


# =============================================================================
# AEV-PLIG Data Preparation
# =============================================================================

rule prepare_aev_plig_data:
    """
    Prepare AEV-PLIG rescoring data for a single target.

    Reads manifest and creates CSV with:
      - Molecular descriptors (MW, LogP, HBD, HBA)
      - Docking scores (Vina affinity, pK)
      - File paths (SDF, protein PDB)
      - Activity labels

    This CSV is used as input for the AEV-PLIG neural network.

    Wildcards:
        dataset: Dataset name (e.g., LIT_PCBA)
        target: Target protein ID (e.g., ADRB2)

    Output:
        - CSV file with rescoring data
    """
    input:
        manifest = MANIFEST_PATH,
        conversion_checkpoint = "data/logs/conversion/conversion_checkpoint.done",

    output:
        csv = "{dataset}/{target}/rescoring/datasets/aev_plig_{target}.csv",

    log:
        "data/logs/rescoring/{dataset}_{target}_prep.log"

    conda:
        "../envs/vscreen.yaml"

    resources:
        mem_mb = lambda wildcards: get_resources('rescoring').get('mem_mb', 32000),
        cpus = lambda wildcards: get_resources('rescoring').get('cpus', 16),
        runtime = lambda wildcards: get_resources('rescoring').get('time_min', 20),

    shell:
        """
        python workflow/scripts/rescore_aev_plig.py \
            --manifest {input.manifest} \
            --target {wildcards.target} \
            --output {output.csv} \
            2>&1 | tee {log}
        """


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


def get_rescoring_csv_paths():
    """
    Get list of AEV-PLIG CSV paths for all targets with docked ligands.

    Returns:
        List of CSV file paths
    """
    targets = get_targets_needing_rescoring()
    dataset = config['dataset']

    csv_paths = [
        f"{dataset}/{target}/rescoring/datasets/aev_plig_{target}.csv"
        for target in targets
    ]

    return csv_paths


# =============================================================================
# Batch Rescoring Rules
# =============================================================================

checkpoint rescoring_checkpoint:
    """
    Checkpoint to determine which targets need AEV-PLIG data preparation.

    Reads manifest and identifies targets that have docked ligands.
    """
    input:
        manifest = MANIFEST_PATH,
        conversion_checkpoint = "data/logs/conversion/conversion_checkpoint.done",

    output:
        touch("data/logs/rescoring/rescoring_checkpoint.done")

    run:
        manifest = load_manifest()

        docked = manifest[manifest['docking_status'] == True]
        converted = manifest[
            (manifest['docking_status'] == True) &
            manifest['docked_sdf_path'].apply(lambda x: Path(x).exists())
        ]

        targets_with_docking = docked['protein_id'].nunique()
        targets_ready = converted['protein_id'].nunique()

        print(f"\nRescoring status:")
        print(f"  Targets with docking: {targets_with_docking}")
        print(f"  Targets with SDF files: {targets_ready}")
        print(f"  Total docked ligands: {len(docked)}")
        print(f"  Ligands with SDF: {len(converted)}")


def get_rescoring_csvs(wildcards):
    """
    Dynamic input function for prepare_all_aev_plig rule.

    Called after checkpoint completes, determines which CSV files to create.
    """
    # Trigger checkpoint
    checkpoints.rescoring_checkpoint.get()

    # Get CSV paths for all targets
    return get_rescoring_csv_paths()


rule prepare_all_aev_plig:
    """
    Prepare AEV-PLIG data for all targets.

    Uses checkpoint to dynamically determine which targets have docked ligands.
    """
    input:
        get_rescoring_csvs

    message:
        "AEV-PLIG data preparation complete for all targets!"


# =============================================================================
# Optional: Run AEV-PLIG Neural Network
# =============================================================================

# NOTE: This rule is commented out because it requires the AEV-PLIG
# neural network model files. Uncomment and configure if you have the model.
#
# rule run_aev_plig_rescoring:
#     """
#     Run AEV-PLIG neural network rescoring on prepared data.
#
#     Requires:
#       - AEV-PLIG model files in AEV-PLIG/ directory
#       - Configured AEV-PLIG environment
#     """
#     input:
#         csv = "{dataset}/{target}/rescoring/datasets/aev_plig_{target}.csv",
#
#     output:
#         scores = "{dataset}/{target}/rescoring/results/aev_plig_{target}_scores.csv",
#
#     log:
#         "data/logs/rescoring/{dataset}_{target}_run.log"
#
#     params:
#         aev_plig_dir = lambda wildcards: get_tool_path('aev_plig_dir'),
#
#     shell:
#         """
#         # Example command (adjust to your AEV-PLIG setup):
#         cd {params.aev_plig_dir} && \
#         python run_aev_plig.py \
#             --input {input.csv} \
#             --output {output.scores} \
#             2>&1 | tee {log}
#         """


# =============================================================================
# Convenience Rules
# =============================================================================

rule rescore_target:
    """
    Prepare AEV-PLIG data for a specific target.

    Usage: snakemake rescore_target --config target=ADRB2
    """
    input:
        lambda wildcards: expand(
            "{dataset}/{target}/rescoring/datasets/aev_plig_{target}.csv",
            dataset=config['dataset'],
            target=config.get('target', 'ADRB2')
        )

    message:
        "Target rescoring data prepared!"


rule rescore_all:
    """
    Complete rescoring stage (data preparation).

    This prepares AEV-PLIG CSV files for all targets with docked ligands.
    """
    input:
        "data/logs/rescoring/rescoring_checkpoint.done"

    message:
        "Rescoring stage complete!"
