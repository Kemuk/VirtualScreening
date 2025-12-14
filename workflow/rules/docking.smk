"""
docking.smk

Snakemake rules for molecular docking using AutoDock Vina (GPU/CPU).

Rules:
  - dock_ligand_gpu: Dock single ligand using GPU
  - dock_ligand_cpu: Dock single ligand using CPU
  - dock_all_gpu: Dock all ligands using GPU
  - dock_all_cpu: Dock all ligands using CPU
"""

import pandas as pd


# =============================================================================
# Rule Order
# =============================================================================
# When both GPU and CPU rules can produce same output, prefer GPU
ruleorder: dock_ligand_gpu > dock_ligand_cpu


# =============================================================================
# GPU Docking
# =============================================================================

rule dock_ligand_gpu:
    """
    Dock a single ligand using GPU-accelerated Vina.

    Input:
        - Receptor PDBQT
        - Ligand PDBQT (prepared)

    Output:
        - Docked PDBQT with multiple binding modes
        - Log file with scores

    Wildcards:
        dataset: Dataset name (e.g., LIT_PCBA)
        target: Target protein ID (e.g., ADRB2)
        ligand_class: 'actives' or 'inactives'
        ligand_id: Ligand identifier
    """
    input:
        receptor = "{dataset}/{target}/{target}_protein.pdbqt",
        ligand = "{dataset}/{target}/pdbqt/{ligand_class}/{ligand_id}.pdbqt",

    output:
        docked = "{dataset}/{target}/docked_vina/{ligand_class}/{ligand_id}_docked.pdbqt",

    log:
        "data/logs/docking/{dataset}_{target}_{ligand_class}_{ligand_id}_gpu.log"

    conda:
        "../envs/vscreen.yaml"

    resources:
        mem_mb = lambda wildcards: get_resources('docking_gpu').get('mem_mb', 20000),
        cpus = lambda wildcards: get_resources('docking_gpu').get('cpus', 2),
        gpus = lambda wildcards: get_resources('docking_gpu').get('gpus', 1),
        runtime = lambda wildcards: get_resources('docking_gpu').get('time_min', 720),

    params:
        vina_bin = lambda wildcards: get_tool_path('vina_gpu'),
        exhaustiveness = lambda wildcards: config.get('docking', {}).get('exhaustiveness', 8),
        num_modes = lambda wildcards: config.get('docking', {}).get('num_modes', 9),
        energy_range = lambda wildcards: config.get('docking', {}).get('energy_range', 3),
        seed = lambda wildcards: config.get('docking', {}).get('seed', 42),
        gpu_threads = lambda wildcards: config.get('gpu', {}).get('threads', 8000),
        box = lambda wildcards: get_box_params_for_ligand(
            wildcards.target,
            wildcards.ligand_id,
        ),

    shell:
        """
        python workflow/scripts/dock_vina.py \
            --receptor {input.receptor} \
            --ligand {input.ligand} \
            --output {output.docked} \
            --center-x {params.box[center_x]} \
            --center-y {params.box[center_y]} \
            --center-z {params.box[center_z]} \
            --size-x {params.box[size_x]} \
            --size-y {params.box[size_y]} \
            --size-z {params.box[size_z]} \
            --vina-bin {params.vina_bin} \
            --exhaustiveness {params.exhaustiveness} \
            --num-modes {params.num_modes} \
            --energy-range {params.energy_range} \
            --seed {params.seed} \
            --gpu-threads {params.gpu_threads} \
            --mode gpu \
            2>&1 | tee {log}
        """


# =============================================================================
# CPU Docking
# =============================================================================

rule dock_ligand_cpu:
    """
    Dock a single ligand using CPU Vina.

    Same as dock_ligand_gpu but uses CPU threads instead of GPU.
    """
    input:
        receptor = "{dataset}/{target}/{target}_protein.pdbqt",
        ligand = "{dataset}/{target}/pdbqt/{ligand_class}/{ligand_id}.pdbqt",

    output:
        docked = "{dataset}/{target}/docked_vina/{ligand_class}/{ligand_id}_docked.pdbqt",

    log:
        "data/logs/docking/{dataset}_{target}_{ligand_class}_{ligand_id}_cpu.log"

    conda:
        "../envs/vscreen.yaml"

    resources:
        mem_mb = lambda wildcards: get_resources('docking_cpu').get('mem_mb', 64000),
        cpus = lambda wildcards: get_resources('docking_cpu').get('cpus', 32),
        runtime = lambda wildcards: get_resources('docking_cpu').get('time_min', 720),

    params:
        vina_bin = lambda wildcards: get_tool_path('vina_cpu'),
        exhaustiveness = lambda wildcards: config.get('docking', {}).get('exhaustiveness', 8),
        num_modes = lambda wildcards: config.get('docking', {}).get('num_modes', 9),
        energy_range = lambda wildcards: config.get('docking', {}).get('energy_range', 3),
        seed = lambda wildcards: config.get('docking', {}).get('seed', 42),
        cpu_threads = lambda wildcards: config.get('cpu', {}).get('threads', 8),
        box = lambda wildcards: get_box_params_for_ligand(
            wildcards.target,
            wildcards.ligand_id,
        ),

    shell:
        """
        python workflow/scripts/dock_vina.py \
            --receptor {input.receptor} \
            --ligand {input.ligand} \
            --output {output.docked} \
            --center-x {params.box[center_x]} \
            --center-y {params.box[center_y]} \
            --center-z {params.box[center_z]} \
            --size-x {params.box[size_x]} \
            --size-y {params.box[size_y]} \
            --size-z {params.box[size_z]} \
            --vina-bin {params.vina_bin} \
            --exhaustiveness {params.exhaustiveness} \
            --num-modes {params.num_modes} \
            --energy-range {params.energy_range} \
            --seed {params.seed} \
            --threads {params.cpu_threads} \
            --mode cpu \
            2>&1 | tee {log}
        """


# =============================================================================
# Helper Functions
# =============================================================================

def get_box_params_for_ligand(target_id: str, ligand_id: str) -> dict:
    """
    Get docking box parameters for a specific ligand.

    Reads from manifest to get box parameters for the target.

    Args:
        target_id: Target protein ID
        ligand_id: Ligand identifier

    Returns:
        dict with keys: center_x, center_y, center_z, size_x, size_y, size_z
    """
    manifest = load_manifest()

    # Find the ligand in manifest
    compound_key = f"{target_id}_{ligand_id}"
    row = manifest[manifest['compound_key'] == compound_key]

    if len(row) == 0:
        # Fallback to target config if not in manifest
        return get_box_params(target_id)

    # Extract box parameters from manifest
    return {
        'center_x': row.iloc[0]['box_center_x'],
        'center_y': row.iloc[0]['box_center_y'],
        'center_z': row.iloc[0]['box_center_z'],
        'size_x': row.iloc[0]['box_size_x'],
        'size_y': row.iloc[0]['box_size_y'],
        'size_z': row.iloc[0]['box_size_z'],
    }


def get_ligands_needing_docking():
    """
    Get list of docked PDBQT paths for ligands that need docking.

    Filters manifest for ligands where:
      - preparation_status = True
      - docking_status = False

    Returns:
        List of docked PDBQT file paths
    """
    manifest = load_manifest()

    # Filter to prepared but undocked ligands
    needs_docking = manifest[
        (manifest['preparation_status'] == True) &
        (manifest['docking_status'] == False)
    ]

    # Return list of docked output paths
    return needs_docking['docked_pdbqt_path'].tolist()


# =============================================================================
# Batch Docking Rules
# =============================================================================

checkpoint docking_checkpoint:
    """
    Checkpoint to determine which ligands need docking.

    Reads manifest and identifies ligands that are prepared but not docked.
    """
    input:
        manifest = MANIFEST_PATH,
        prep_checkpoint = "data/logs/preparation/ligands_checkpoint.done",

    output:
        touch("data/logs/docking/docking_checkpoint.done")

    run:
        manifest = load_manifest()

        prepared = manifest[manifest['preparation_status'] == True]
        docked = manifest[manifest['docking_status'] == True]
        needs_docking = manifest[
            (manifest['preparation_status'] == True) &
            (manifest['docking_status'] == False)
        ]

        print(f"\nDocking status:")
        print(f"  Prepared ligands: {len(prepared)}")
        print(f"  Already docked: {len(docked)}")
        print(f"  Need docking: {len(needs_docking)}")


def get_docked_ligands(wildcards):
    """
    Dynamic input function for dock_all rules.

    Called after checkpoint completes, determines which ligands to dock.
    """
    # Trigger checkpoint
    checkpoints.docking_checkpoint.get()

    # Get ligands needing docking
    return get_ligands_needing_docking()


rule dock_all_gpu:
    """
    Dock all prepared ligands using GPU.

    Uses checkpoint to dynamically determine which ligands need docking.
    """
    input:
        get_docked_ligands

    message:
        "GPU docking complete for all ligands!"


rule dock_all_cpu:
    """
    Dock all prepared ligands using CPU.

    Uses checkpoint to dynamically determine which ligands need docking.
    """
    input:
        get_docked_ligands

    message:
        "CPU docking complete for all ligands!"


rule dock_all:
    """
    Dock all prepared ligands (mode determined by config or user choice).

    This is an alias that can be configured to use GPU or CPU.
    """
    input:
        "data/logs/docking/docking_checkpoint.done"

    message:
        "Docking stage complete!"
