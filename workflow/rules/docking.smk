"""
docking.smk

Snakemake rules for molecular docking using AutoDock Vina (GPU/CPU).

Docking mode is controlled by config['docking']['mode']:
  - 'gpu': Use GPU-accelerated Vina (default)
  - 'cpu': Use CPU-based Vina

Rules:
  - dock_ligand_gpu: Dock single ligand using GPU
  - dock_ligand_cpu: Dock single ligand using CPU
  - dock_all_gpu: Dock all ligands using GPU
  - dock_all_cpu: Dock all ligands using CPU
  - dock_all: Dock all ligands using configured mode
"""

# =============================================================================
# Configuration
# =============================================================================

# Docking mode from config (gpu or cpu)
DOCKING_MODE = config.get('docking', {}).get('mode', 'gpu')
DOCK_CHUNKS = get_chunk_count("gpu")
DOCK_CHUNK_IDS = list(range(DOCK_CHUNKS))


# =============================================================================
# Rule Order
# =============================================================================
# When both GPU and CPU rules can produce same output, prefer based on config
if DOCKING_MODE == 'gpu':
    ruleorder: dock_ligand_gpu > dock_ligand_cpu
else:
    ruleorder: dock_ligand_cpu > dock_ligand_gpu


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
            --progress \
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
# Batch Docking Rules
# =============================================================================

rule shard_docking:
    """Shard ligands needing docking into chunk CSVs."""
    input:
        manifest = MANIFEST_PATH,
        prep_checkpoint = "data/logs/preparation/ligands_checkpoint.done",

    output:
        expand("data/chunks/docking/chunk_{chunk}.csv", chunk=DOCK_CHUNK_IDS)

    log:
        "data/logs/docking/shard_docking.log"

    params:
        num_chunks = DOCK_CHUNKS,
        include_done = "--include-done" if config.get("mode", "production") == "devel" else "",

    conda:
        "../envs/vscreen.yaml"

    shell:
        """
        python workflow/scripts/shard_stage.py \
            --stage docking \
            --manifest {input.manifest} \
            --outdir data/chunks/docking \
            --num-chunks {params.num_chunks} \
            {params.include_done} \
            2>&1 | tee {log}
        """


rule dock_chunk:
    """Dock ligands for a single chunk."""
    input:
        chunk = "data/chunks/docking/chunk_{chunk}.csv"

    output:
        results = "data/results/docking/chunk_{chunk}.csv"

    log:
        "data/logs/docking/dock_chunk_{chunk}.log"

    conda:
        "../envs/vscreen.yaml"

    shell:
        """
        python workflow/scripts/process_stage_chunk.py \
            --stage docking \
            --chunk {input.chunk} \
            --output {output.results} \
            2>&1 | tee {log}
        """


rule merge_docking_results:
    """Merge docking chunk results into the manifest."""
    input:
        manifest = MANIFEST_PATH,
        results = expand("data/results/docking/chunk_{chunk}.csv", chunk=DOCK_CHUNK_IDS),

    output:
        touch("data/logs/docking/docking_checkpoint.done")

    log:
        "data/logs/docking/merge_docking_results.log"

    conda:
        "../envs/vscreen.yaml"

    shell:
        """
        python workflow/scripts/merge_stage_results.py \
            --stage docking \
            --manifest {input.manifest} \
            --results-dir data/results/docking \
            2>&1 | tee {log}
        """


rule dock_all:
    """
    Dock all prepared ligands using the configured mode.

    Mode is determined by config['docking']['mode']:
      - 'gpu': Uses GPU-accelerated Vina
      - 'cpu': Uses CPU-based Vina
    """
    input:
        "data/logs/docking/docking_checkpoint.done"

    message:
        f"Docking complete using {DOCKING_MODE.upper()} mode!"


# =============================================================================
# Docking Mode Info
# =============================================================================

rule docking_info:
    """
    Print current docking configuration.
    """
    run:
        print(f"\nDocking Configuration:")
        print(f"  Mode: {DOCKING_MODE}")
        print(f"  Vina binary: {config.get('tools', {}).get(f'vina_{DOCKING_MODE}', 'N/A')}")
        print(f"  Exhaustiveness: {config.get('docking', {}).get('exhaustiveness', 8)}")
        print(f"  Num modes: {config.get('docking', {}).get('num_modes', 9)}")
        if DOCKING_MODE == 'gpu':
            print(f"  GPU threads: {config.get('gpu', {}).get('threads', 8000)}")
        else:
            print(f"  CPU threads: {config.get('cpu', {}).get('threads', 8)}")
