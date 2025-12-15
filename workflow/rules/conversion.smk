"""
conversion.smk

Snakemake rules for post-docking format conversion (PDBQT → SDF).

SDF format is required for:
  - Visualization in molecular viewers
  - AEV-PLIG rescoring
  - General downstream analysis

Rules:
  - convert_to_sdf: Convert single docked PDBQT to SDF
  - convert_all_to_sdf: Batch conversion of all docked ligands
"""

import pandas as pd


# =============================================================================
# Single Conversion Rule
# =============================================================================

rule convert_to_sdf:
    """
    Convert a single docked PDBQT file to SDF format.

    Extracts the best binding mode (model 0) by default and converts
    to SDF using OpenBabel.

    Wildcards:
        dataset: Dataset name (e.g., LIT_PCBA)
        target: Target protein ID (e.g., ADRB2)
        ligand_class: 'actives' or 'inactives'
        ligand_id: Ligand identifier

    Input:
        - Docked PDBQT file (with multiple binding modes)

    Output:
        - SDF file (single binding mode)
    """
    input:
        pdbqt = "{dataset}/{target}/docked_vina/{ligand_class}/{ligand_id}_docked.pdbqt",

    output:
        sdf = "{dataset}/{target}/docked_sdf/{ligand_class}/{ligand_id}.sdf",

    log:
        "data/logs/conversion/{dataset}_{target}_{ligand_class}_{ligand_id}.log"

    conda:
        "../envs/vscreen.yaml"

    resources:
        mem_mb = lambda wildcards: get_resources('sdf_conversion').get('mem_mb', 8000),
        cpus = lambda wildcards: get_resources('sdf_conversion').get('cpus', 4),
        runtime = lambda wildcards: get_resources('sdf_conversion').get('time_min', 30),

    params:
        model_index = lambda wildcards: config.get('sdf_conversion', {}).get('select_model', 0),

    shell:
        """
        python workflow/scripts/pdbqt_to_sdf.py \
            --input {input.pdbqt} \
            --output {output.sdf} \
            --model {params.model_index} \
            --ligand-id {wildcards.ligand_id} \
            2>&1 | tee {log}
        """


# =============================================================================
# Helper Functions
# =============================================================================

def get_ligands_needing_conversion():
    """
    Get list of SDF paths for ligands that need conversion.

    Filters manifest for ligands where:
      - docking_status = True
      - SDF file doesn't exist yet

    Returns:
        List of SDF file paths
    """
    manifest = load_manifest()

    # Filter to docked ligands
    docked = manifest[manifest['docking_status'] == True]

    # Check which SDF files don't exist yet
    needs_conversion = []
    for _, row in docked.iterrows():
        sdf_path = Path(row['docked_sdf_path'])
        if not sdf_path.exists():
            needs_conversion.append(str(sdf_path))

    return needs_conversion


# =============================================================================
# Batch Conversion Rules
# =============================================================================

checkpoint conversion_checkpoint:
    """
    Checkpoint to determine which docked ligands need SDF conversion.

    Reads manifest and identifies ligands that have been docked but
    not yet converted to SDF.
    """
    input:
        manifest = MANIFEST_PATH,
        docking_checkpoint = "data/logs/docking/docking_checkpoint.done",

    output:
        touch("data/logs/conversion/conversion_checkpoint.done")

    run:
        manifest = load_manifest()

        docked = manifest[manifest['docking_status'] == True]

        # Count existing SDF files
        existing_sdfs = sum(1 for _, row in docked.iterrows()
                          if Path(row['docked_sdf_path']).exists())
        needs_conversion = len(docked) - existing_sdfs

        print(f"\nConversion status:")
        print(f"  Docked ligands: {len(docked)}")
        print(f"  Already converted: {existing_sdfs}")
        print(f"  Need conversion: {needs_conversion}")


def get_converted_ligands(wildcards):
    """
    Dynamic input function for convert_all_to_sdf rule.

    Called after checkpoint completes, determines which SDF files to create.
    """
    # Trigger checkpoint
    checkpoints.conversion_checkpoint.get()

    # Get ligands needing conversion
    return get_ligands_needing_conversion()


rule convert_all_to_sdf:
    """
    Convert all docked ligands to SDF format.

    Uses checkpoint to dynamically determine which ligands need conversion.
    """
    input:
        get_converted_ligands

    message:
        "SDF conversion complete for all docked ligands!"


# =============================================================================
# Convenience Rules
# =============================================================================

rule convert_target_to_sdf:
    """
    Convert all docked ligands for a specific target to SDF.

    Usage: snakemake convert_target_to_sdf --config target=ADRB2
    """
    input:
        lambda wildcards: expand(
            "{dataset}/{target}/docked_sdf/{ligand_class}/{ligand_id}.sdf",
            dataset=config['dataset'],
            target=config.get('target', 'ADRB2'),
            ligand_class=['actives', 'inactives'],
            ligand_id=get_ligand_ids_for_target(config.get('target', 'ADRB2'))
        )

    message:
        "Target SDF conversion complete!"


def get_ligand_ids_for_target(target_id: str) -> list:
    """
    Get list of ligand IDs for a specific target from manifest.

    Args:
        target_id: Target protein ID

    Returns:
        List of ligand IDs
    """
    manifest = load_manifest()
    target_ligands = manifest[manifest['protein_id'] == target_id]
    return target_ligands['ligand_id'].unique().tolist()
