"""
preparation.smk

Snakemake rules for receptor and ligand preparation.

Simplified rules after consolidation:
  - prepare_receptor: Prepare single receptor (development/testing)
  - prepare_all: Prepare all receptors and ligands via SLURM (production)
"""

MODE = config.get('mode', 'production')


# =============================================================================
# Single-file Preparation (Development/Testing)
# =============================================================================

rule prepare_receptor:
    """
    Convert receptor MOL2 → PDBQT + PDB (for development/testing).

    For production preparation of all receptors, use: snakemake prepare_all
    """
    input:
        mol2 = lambda wildcards: get_target_config(wildcards.target)['receptor_mol2']

    output:
        pdbqt = "{dataset}/{target}/{target}_protein.pdbqt",
        pdb = "{dataset}/{target}/{target}_protein.pdb"

    log:
        "data/logs/preparation/{dataset}_{target}_receptor.log"

    conda:
        "../envs/vscreen.yaml"

    params:
        ph = lambda wildcards: config.get('preparation', {}).get('ph', 7.4),
        partial_charge = lambda wildcards: config.get('preparation', {}).get('partial_charge', 'gasteiger'),

    shell:
        """
        python workflow/scripts/mol2_to_pdbqt.py \
            --input {input.mol2} \
            --pdbqt {output.pdbqt} \
            --pdb {output.pdb} \
            --ph {params.ph} \
            --partial-charge {params.partial_charge} \
            --overwrite \
            2>&1 | tee {log}
        """


# =============================================================================
# Production Preparation (SLURM Array Jobs)
# =============================================================================

rule prepare_all:
    """
    Prepare all receptors and ligands via SLURM array jobs.

    This is the recommended production method for preparing large numbers of ligands.
    Uses the unified submit_stage.py to handle SLURM job submission.

    Includes:
      - Receptor preparation (MOL2 → PDBQT + PDB)
      - Ligand preparation (SMILES → PDBQT)
    """
    input:
        manifest = MANIFEST_PATH

    output:
        checkpoint = touch("data/logs/preparation/preparation_checkpoint.done")

    log:
        "data/logs/preparation/prepare_all.log"

    conda:
        "../envs/vscreen.yaml"

    params:
        mode = MODE,

    shell:
        """
        # First ensure all receptors are prepared
        echo "Preparing receptors..."
        snakemake --cores 1 \
            $(snakemake --list-targets-rules prepare_receptor 2>/dev/null || echo "") \
            2>&1 | tee -a {log}

        # Then prepare ligands via SLURM
        echo "Submitting ligand preparation jobs..."
        python -m workflow.slurm.submit_stage \
            --stage ligands \
            --mode {params.mode} \
            --manifest {input.manifest} \
            2>&1 | tee -a {log}
        """
