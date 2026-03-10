"""
docking.smk

Snakemake rules for molecular docking using AutoDock Vina (GPU/CPU).

Simplified rules after consolidation:
  - dock_ligand: Dock single ligand (development/testing)
  - dock_all: Dock all ligands via SLURM array jobs (production)

Docking mode is controlled by config['docking']['mode']:
  - 'gpu': Use GPU-accelerated Vina (default)
  - 'cpu': Use CPU-based Vina
"""

# =============================================================================
# Configuration
# =============================================================================

DOCKING_MODE = config.get('docking', {}).get('mode', 'gpu')
MODE = config.get('mode', 'production')



# =============================================================================
# Single-file Docking (Development/Testing)
# =============================================================================

rule dock_ligand:
    """
    Dock a single ligand (for development/testing, not production).

    Unified rule that works for both GPU and CPU modes.
    Mode determined by config['docking']['mode'] (gpu or cpu).

    For production docking of all ligands, use: snakemake dock_all
    """
    input:
        receptor = "{dataset}/{target}/{target}_protein.pdbqt",
        ligand = "{dataset}/{target}/pdbqt/{ligand_class}/{ligand_id}.pdbqt",

    output:
        docked = "{dataset}/{target}/docked_vina/{ligand_class}/{ligand_id}_docked.pdbqt",

    log:
        "data/logs/docking/{dataset}_{target}_{ligand_class}_{ligand_id}.log"

    conda:
        "../envs/vscreen.yaml"

    resources:
        mem_mb = lambda wildcards: get_resources(f'docking_{DOCKING_MODE}').get('mem_mb', 20000),
        cpus = lambda wildcards: get_resources(f'docking_{DOCKING_MODE}').get('cpus', 2),
        gpus = lambda wildcards: get_resources(f'docking_{DOCKING_MODE}').get('gpus', 0) if DOCKING_MODE == 'gpu' else 0,
        runtime = lambda wildcards: get_resources(f'docking_{DOCKING_MODE}').get('time_min', 720),

    params:
        mode = DOCKING_MODE,
        vina_bin = lambda wildcards: get_tool_path(f'vina_{DOCKING_MODE}'),
        exhaustiveness = lambda wildcards: config.get('docking', {}).get('exhaustiveness', 8),
        num_modes = lambda wildcards: config.get('docking', {}).get('num_modes', 9),
        energy_range = lambda wildcards: config.get('docking', {}).get('energy_range', 3),
        seed = lambda wildcards: config.get('docking', {}).get('seed', 42),
        threads_or_gpu = lambda wildcards: (
            config.get('gpu', {}).get('threads', 8000) if DOCKING_MODE == 'gpu'
            else config.get('cpu', {}).get('threads', 8)
        ),
        box = lambda wildcards: get_box_params_for_ligand(
            wildcards.target,
            wildcards.ligand_id,
        ),

    shell:
        """
        # Load modules if needed for GPU
        if [ "{params.mode}" = "gpu" ]; then
            module load Boost/1.77.0-GCC-11.2.0 CUDA/12.0.0 2>/dev/null || true
        fi

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
            {"--gpu-threads" if params.mode == "gpu" else "--threads"} {params.threads_or_gpu} \
            --mode {params.mode} \
            --progress \
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


# =============================================================================
# Production Docking (SLURM Array Jobs)
# =============================================================================

rule dock_all:
    """
    Dock all prepared ligands via SLURM array jobs.

    This is the recommended production method for docking large numbers of ligands.
    Uses the unified submit_stage.py to handle SLURM job submission.

    Mode is determined by config['docking']['mode']:
      - 'gpu': Uses GPU-accelerated Vina
      - 'cpu': Uses CPU-based Vina
    """
    input:
        manifest = MANIFEST_PATH,
        receptors = lambda wildcards: expand(
            "{dataset}/{target}/{target}_protein.pdbqt",
            dataset=config['dataset'],
            target=get_targets()
        ),

    output:
        checkpoint = touch("data/logs/docking/docking_checkpoint.done")

    log:
        "data/logs/docking/dock_all.log"

    conda:
        "../envs/vscreen.yaml"

    params:
        mode = MODE,
        docking_mode = DOCKING_MODE,

    shell:
        """
        python -m workflow.slurm.submit_stage \
            --stage docking \
            --mode {params.mode} \
            --docking-mode {params.docking_mode} \
            --manifest {input.manifest} \
            2>&1 | tee {log}
        """
