"""
conversion.smk

Snakemake rules for post-docking format conversion (PDBQT → SDF).

Simplified rules after consolidation:
  - convert_to_sdf: Convert single file (development/testing)
  - convert_all: Convert all via SLURM array jobs (production)

SDF format is required for:
  - Visualization in molecular viewers
  - AEV-PLIG rescoring
  - General downstream analysis
"""

MODE = config.get('mode', 'production')


# =============================================================================
# Single-file Conversion (Development/Testing)
# =============================================================================

rule convert_to_sdf:
    """
    Convert a single docked PDBQT file to SDF format (for development/testing).

    Extracts the best binding mode (model 0) by default and converts
    to SDF using OpenBabel.

    For production conversion of all ligands, use: snakemake convert_all
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
# Production Conversion (SLURM Array Jobs)
# =============================================================================

rule convert_all:
    """
    Convert all docked ligands to SDF format via SLURM array jobs.

    This is the recommended production method for converting large numbers of ligands.
    Uses the unified submit_stage.py to handle SLURM job submission.
    """
    input:
        manifest = MANIFEST_PATH,
        docking_checkpoint = "data/logs/docking/docking_checkpoint.done",

    output:
        checkpoint = touch("data/logs/conversion/conversion_checkpoint.done")

    log:
        "data/logs/conversion/convert_all.log"

    conda:
        "../envs/vscreen.yaml"

    params:
        mode = MODE,

    shell:
        """
        python -m workflow.slurm.submit_stage \
            --stage conversion \
            --mode {params.mode} \
            --manifest {input.manifest} \
            2>&1 | tee {log}
        """
