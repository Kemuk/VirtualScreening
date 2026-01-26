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
    },
    'docking': {
        'status_column': 'docking_status',
        'depends_on': 'preparation_status',
        'score_column': 'vina_score',
        'worker_module': 'workflow.slurm.workers.docking',
        'description': 'Vina GPU/CPU docking',
        'check_file_column': 'docked_pdbqt_path',
    },
    'conversion': {
        'status_column': 'conversion_status',
        'depends_on': 'docking_status',
        'score_column': None,
        'worker_module': 'workflow.slurm.workers.conversion',
        'description': 'PDBQT to SDF conversion',
        # For conversion, we can also check file existence if needed
        'check_file_column': 'docked_sdf_path',
    },
    'aev_infer': {
        'status_column': 'rescoring_status',
        'depends_on': 'docking_status',
        'score_column': 'aev_plig_best_score',
        'worker_module': 'workflow.slurm.workers.aev_infer',
        'description': 'AEV-PLIG neural network rescoring',
    },
}


def get_stage_config(stage: str) -> dict:
    """Get configuration for a stage."""
    if stage not in STAGES:
        valid = ', '.join(STAGES.keys())
        raise ValueError(f"Unknown stage: {stage}. Valid stages: {valid}")
    return STAGES[stage]


def list_stages() -> list:
    """List all available stages."""
    return list(STAGES.keys())
