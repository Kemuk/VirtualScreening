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

CONVERSION_CHUNKS = get_chunk_count("cpu")
CONVERSION_CHUNK_IDS = list(range(CONVERSION_CHUNKS))


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

# =============================================================================
# Batch Conversion Rules
# =============================================================================

rule shard_conversion:
    """Shard docked ligands needing conversion into chunk CSVs."""
    input:
        manifest = MANIFEST_PATH,
        docking_checkpoint = "data/logs/docking/docking_checkpoint.done",

    output:
        expand("data/chunks/conversion/chunk_{chunk}.csv", chunk=CONVERSION_CHUNK_IDS)

    log:
        "data/logs/conversion/shard_conversion.log"

    params:
        num_chunks = CONVERSION_CHUNKS,

    conda:
        "../envs/vscreen.yaml"

    shell:
        """
        python workflow/scripts/shard_stage.py \
            --stage conversion \
            --manifest {input.manifest} \
            --outdir data/chunks/conversion \
            --num-chunks {params.num_chunks} \
            2>&1 | tee {log}
        """


rule convert_chunk_to_sdf:
    """Convert docked ligands to SDF for a single chunk."""
    input:
        chunk = "data/chunks/conversion/chunk_{chunk}.csv"

    output:
        results = "data/results/conversion/chunk_{chunk}.csv"

    log:
        "data/logs/conversion/convert_chunk_{chunk}.log"

    conda:
        "../envs/vscreen.yaml"

    shell:
        """
        python workflow/scripts/process_stage_chunk.py \
            --stage conversion \
            --chunk {input.chunk} \
            --output {output.results} \
            2>&1 | tee {log}
        """


rule merge_conversion_results:
    """Merge conversion chunk results into the manifest."""
    input:
        manifest = MANIFEST_PATH,
        results = expand("data/results/conversion/chunk_{chunk}.csv", chunk=CONVERSION_CHUNK_IDS),

    output:
        touch("data/logs/conversion/conversion_checkpoint.done")

    log:
        "data/logs/conversion/merge_conversion_results.log"

    conda:
        "../envs/vscreen.yaml"

    shell:
        """
        python workflow/scripts/merge_stage_results.py \
            --stage conversion \
            --manifest {input.manifest} \
            --results-dir data/results/conversion \
            2>&1 | tee {log}
        """


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
