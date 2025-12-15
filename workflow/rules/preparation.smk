"""
preparation.smk

Snakemake rules for receptor and ligand preparation.
"""

import pandas as pd


# Receptor Preparation
rule prepare_receptor:
    """Convert receptor MOL2 → PDBQT + PDB."""
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


rule prepare_all_receptors:
    """Prepare all receptors defined in targets.yaml."""
    input:
        expand(
            "{dataset}/{target}/{target}_protein.pdbqt",
            dataset=config['dataset'],
            target=get_targets()
        ),
        expand(
            "{dataset}/{target}/{target}_protein.pdb",
            dataset=config['dataset'],
            target=get_targets()
        )

    message:
        "All receptors prepared!"


# Ligand Preparation (Batch Processing)
rule prepare_all_ligands:
    """Prepare all ligands using batch parallel processing (single progress bar)."""
    input:
        manifest = MANIFEST_PATH

    output:
        touch("data/logs/preparation/ligands_prepared.done")

    log:
        "data/logs/preparation/batch_ligand_preparation.log"

    conda:
        "../envs/vscreen.yaml"

    params:
        ph = config.get('preparation', {}).get('ph', 7.4),
        partial_charge = config.get('preparation', {}).get('partial_charge', 'gasteiger'),
        max_workers_flag = lambda wildcards: f"--max-workers {config.get('preparation', {}).get('max_workers')}" if config.get('preparation', {}).get('max_workers') else "",

    shell:
        """
        python workflow/scripts/prepare_all_ligands.py \
            --manifest {input.manifest} \
            --project-root . \
            --ph {params.ph} \
            --partial-charge {params.partial_charge} \
            {params.max_workers_flag} \
            2>&1 | tee {log}
        """


# Combined Preparation Rule
rule prepare_all:
    """Prepare all receptors and ligands."""
    input:
        rules.prepare_all_receptors.input,
        "data/logs/preparation/ligands_prepared.done"

    message:
        "Preparation stage complete!"
