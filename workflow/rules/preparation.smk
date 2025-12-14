"""
preparation.smk

Snakemake rules for receptor and ligand preparation.

Rules:
  - prepare_receptor: Convert receptor MOL2 → PDBQT + PDB
  - prepare_ligand: Convert ligand SMILES → PDBQT
  - prepare_all_receptors: Prepare all receptors
  - prepare_all_ligands: Prepare all ligands
"""

import pandas as pd


# =============================================================================
# Receptor Preparation
# =============================================================================

rule prepare_receptor:
    """
    Convert a single receptor from MOL2 to PDBQT and PDB formats.

    Input:
        - {target}_protein.mol2

    Output:
        - {target}_protein.pdbqt (for docking)
        - {target}_protein.pdb (for visualization/rescoring)
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
            2>&1 | tee {log}
        """


rule prepare_all_receptors:
    """
    Prepare all receptors defined in targets.yaml.

    This is a convenience rule to prepare all receptors at once.
    """
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


# =============================================================================
# Ligand Preparation
# =============================================================================

rule prepare_ligand:
    """
    Convert a single ligand from SMILES to PDBQT format.

    This rule is triggered per-ligand based on the manifest.
    The SMILES is read from the manifest using the compound_key.

    Wildcards:
        target: Protein target ID (e.g., ADRB2)
        ligand_class: Either 'actives' or 'inactives'
        ligand_id: Ligand identifier

    Output:
        - {dataset}/{target}/pdbqt/{ligand_class}/{ligand_id}.pdbqt
    """
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
            2>&1 | tee {log}
        """


# =============================================================================
# Helper Functions for Ligand Preparation
# =============================================================================

def get_smiles_for_ligand(target_id: str, ligand_id: str, ligand_class: str) -> str:
    """
    Retrieve SMILES string for a ligand from the manifest.

    Args:
        target_id: Target protein ID
        ligand_id: Ligand identifier
        ligand_class: 'actives' or 'inactives'

    Returns:
        SMILES string
    """
    manifest = load_manifest()

    # Construct compound key
    compound_key = f"{target_id}_{ligand_id}"

    # Filter manifest
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


def get_all_ligand_pdbqts_from_manifest():
    """
    Get list of all ligand PDBQT paths that need to be prepared.

    Reads manifest and returns paths for ligands where preparation_status=False.

    Returns:
        List of PDBQT file paths
    """
    manifest = load_manifest()

    # Filter to ligands that need preparation
    needs_prep = manifest[~manifest['preparation_status']]

    # Return list of PDBQT paths
    return needs_prep['ligand_pdbqt_path'].tolist()


# =============================================================================
# Batch Preparation Rules
# =============================================================================

checkpoint prepare_ligands_checkpoint:
    """
    Checkpoint to determine which ligands need preparation.

    This checkpoint reads the manifest and creates a target list
    for Snakemake to process.
    """
    input:
        manifest = MANIFEST_PATH

    output:
        touch("data/logs/preparation/ligands_checkpoint.done")

    run:
        manifest = load_manifest()
        needs_prep = manifest[~manifest['preparation_status']]

        print(f"\nLigands needing preparation: {len(needs_prep)}")
        print(f"  Actives: {needs_prep['is_active'].sum()}")
        print(f"  Inactives: {(~needs_prep['is_active']).sum()}")


def get_prepared_ligands(wildcards):
    """
    Dynamic input function for prepare_all_ligands rule.

    This is called after the checkpoint completes and determines
    which ligand files need to be created.
    """
    # Trigger checkpoint
    checkpoints.prepare_ligands_checkpoint.get()

    # Load manifest
    manifest = load_manifest()

    # Get ligands that need preparation
    needs_prep = manifest[~manifest['preparation_status']]

    # Return list of output paths
    return needs_prep['ligand_pdbqt_path'].tolist()


rule prepare_all_ligands:
    """
    Prepare all ligands based on manifest.

    Uses checkpoint to dynamically determine which ligands need preparation.
    """
    input:
        get_prepared_ligands

    message:
        "All ligands prepared!"


# =============================================================================
# Combined Preparation Rule
# =============================================================================

rule prepare_all:
    """
    Prepare all receptors and ligands.

    This is the main preparation target.
    """
    input:
        rules.prepare_all_receptors.input,
        "data/logs/preparation/ligands_checkpoint.done"

    message:
        "Preparation stage complete!"
