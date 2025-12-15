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


# Ligand Preparation
rule prepare_ligand:
    """Convert ligand SMILES → PDBQT."""
    output:
        pdbqt = "{dataset}/{target}/pdbqt/{ligand_class}/{ligand_id}.pdbqt"

    log:
        "data/logs/preparation/{dataset}_{target}_{ligand_class}_{ligand_id}.log"

    conda:
        "../envs/vscreen.yaml"

    params:
        ph = lambda wildcards: config.get('preparation', {}).get('ph', 7.4),
        partial_charge = lambda wildcards: config.get('preparation', {}).get('partial_charge', 'gasteiger'),
        smiles = lambda wildcards: get_smiles_for_ligand(
            wildcards.target,
            wildcards.ligand_id,
            wildcards.ligand_class
        ),

    shell:
        """
        python workflow/scripts/smi2pdbqt.py \
            --smiles "{params.smiles}" \
            --output {output.pdbqt} \
            --ligand-id {wildcards.ligand_id} \
            --ph {params.ph} \
            --partial-charge {params.partial_charge} \
            --progress \
            --overwrite \
            --quiet \
            2>&1 | tee {log}
        """


def get_smiles_for_ligand(target_id: str, ligand_id: str, ligand_class: str) -> str:
    """Retrieve SMILES string for a ligand from the manifest."""
    manifest = load_manifest()
    compound_key = f"{target_id}_{ligand_id}"
    row = manifest[manifest['compound_key'] == compound_key]

    if len(row) == 0:
        raise ValueError(f"Ligand not found in manifest: {compound_key}")

    # Verify ligand class matches
    is_active = (ligand_class == 'actives')
    if row.iloc[0]['is_active'] != is_active:
        raise ValueError(
            f"Ligand class mismatch for {compound_key}: "
            f"expected is_active={is_active}, got {row.iloc[0]['is_active']}"
        )

    return row.iloc[0]['smiles_input']


# Batch Preparation
checkpoint prepare_ligands_checkpoint:
    """Checkpoint to determine which ligands need preparation."""
    input:
        manifest = MANIFEST_PATH

    output:
        touch("data/logs/preparation/ligands_checkpoint.done")

    run:
        manifest = load_manifest()
        print(f"\nTotal ligands in manifest: {len(manifest)}")
        print(f"  Actives: {manifest['is_active'].sum()}")
        print(f"  Inactives: {(~manifest['is_active']).sum()}")


def get_prepared_ligands(wildcards):
    """Dynamic input function - returns all ligand PDBQT paths from manifest."""
    checkpoints.prepare_ligands_checkpoint.get()
    manifest = load_manifest()
    return manifest['ligand_pdbqt_path'].tolist()


rule prepare_all_ligands:
    """Prepare all ligands based on manifest."""
    input:
        get_prepared_ligands

    message:
        "All ligands prepared!"


rule prepare_all:
    """Prepare all receptors and ligands."""
    input:
        rules.prepare_all_receptors.input,
        "data/logs/preparation/ligands_checkpoint.done"

    message:
        "Preparation stage complete!"
