"""
stage_config.py

Stage definitions for the chunked SLURM pipeline.

Each stage that processes per-ligand data (potentially millions of items)
uses the same pattern:
  1. prepare_stage.py filters manifest → pending/{stage}.parquet
  2. Workers read their slice, process, write results/{stage}_{task_id}.csv
  3. update_manifest.py merges results → updates manifest
"""

# Stage configurations for the 4 chunked stages
STAGES = {
    'ligands': {
        'status_column': 'preparation_status',
        'depends_on': None,
        'score_column': None,  # No score for ligand prep
        'worker_module': 'workflow.slurm.workers.ligands',
        'description': 'SMILES to PDBQT conversion',
        'slurm_template': 'ligands.slurm',
        'cluster': 'arc',
        'resources': {
            'production': {
                'partition': 'short',
                'time': '01:00:00',
                'mem': '4G',
                'cpus': 1,
                'max_concurrent': 100,
            },
            'devel': {
                'partition': 'devel',
                'time': '00:10:00',
                'mem': '2G',
                'cpus': 1,
                'max_concurrent': 10,
            },
        },
        'chunk_size': 1000,
    },
    'docking': {
        'status_column': 'docking_status',
        'depends_on': 'preparation_status',
        'score_column': 'vina_score',
        'worker_module': 'workflow.slurm.workers.docking',
        'description': 'Vina GPU/CPU docking',
        'check_file_column': 'docked_pdbqt_path',
        'slurm_template': 'docking.slurm',
        'cluster': 'htc',
        'resources': {
            'production': {
                'gpu': {
                    'time': '02:00:00',
                    'mem': '32G',
                    'cpus': 8,
                    'gpus': 1,
                    'partition': 'gpu',
                    'max_concurrent': 20,
                },
                'cpu': {
                    'time': '04:00:00',
                    'mem': '16G',
                    'cpus': 16,
                    'max_concurrent': 50,
                },
            },
            'devel': {
                'gpu': {
                    'partition': 'devel',
                    'time': '00:10:00',
                    'mem': '16G',
                    'cpus': 4,
                    'gpus': 1,
                    'max_concurrent': 5,
                },
                'cpu': {
                    'partition': 'devel',
                    'time': '00:10:00',
                    'mem': '8G',
                    'cpus': 8,
                    'max_concurrent': 10,
                },
            },
        },
        'chunk_size': 500,
    },
    'conversion': {
        'status_column': 'conversion_status',
        'depends_on': 'docking_status',
        'score_column': None,
        'worker_module': 'workflow.slurm.workers.conversion',
        'description': 'PDBQT to SDF conversion',
        'check_file_column': 'docked_sdf_path',
        'slurm_template': 'conversion.slurm',
        'cluster': 'arc',
        'resources': {
            'production': {
                'time': '01:00:00',
                'mem': '4G',
                'cpus': 1,
                'max_concurrent': 100,
            },
            'devel': {
                'partition': 'devel',
                'time': '00:10:00',
                'mem': '2G',
                'cpus': 1,
                'max_concurrent': 10,
            },
        },
        'chunk_size': 1000,
    },
    'aev_infer': {
        'status_column': 'rescoring_status',
        'depends_on': 'docking_status',
        'score_column': 'aev_plig_best_score',
        'worker_module': 'workflow.slurm.workers.aev_infer',
        'description': 'AEV-PLIG neural network rescoring',
        'slurm_template': 'aev_infer.slurm',
        'cluster': 'htc',
        'resources': {
            'production': {
                'time': '02:00:00',
                'mem': '16G',
                'cpus': 4,
                'gpus': 1,
                'partition': 'short',
                'max_concurrent': 20,
            },
            'devel': {
                'partition': 'devel',
                'time': '00:10:00',
                'mem': '8G',
                'cpus': 2,
                'gpus': 1,
                'max_concurrent': 5,
            },
        },
        'chunk_size': 500,
    },
}


def get_stage_config(stage: str) -> dict:
    """Get configuration for a stage."""
    if stage not in STAGES:
        valid = ', '.join(STAGES.keys())
        raise ValueError(f"Unknown stage: {stage}. Valid stages: {valid}")
    return STAGES[stage]


def get_stage_resources(stage: str, mode: str = 'production', docking_mode: str = 'gpu') -> dict:
    """
    Get resource configuration for a stage.

    Args:
        stage: Stage name
        mode: 'production' or 'devel'
        docking_mode: 'gpu' or 'cpu' (only applies to docking stage)

    Returns:
        dict: Resource configuration with keys: time, mem, cpus, max_concurrent,
              and optionally: partition, gpus
    """
    config = get_stage_config(stage)

    if mode not in ('production', 'devel'):
        raise ValueError(f"Invalid mode: {mode}. Must be 'production' or 'devel'")

    resources = config['resources'][mode]

    # For docking stage, handle GPU/CPU mode
    if stage == 'docking' and isinstance(resources, dict) and 'gpu' in resources:
        if docking_mode not in ('gpu', 'cpu'):
            raise ValueError(f"Invalid docking_mode: {docking_mode}. Must be 'gpu' or 'cpu'")
        resources = resources[docking_mode]

    return resources


def list_stages() -> list:
    """List all available stages."""
    return list(STAGES.keys())
