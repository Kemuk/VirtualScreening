"""
SLURM Array Job Orchestrator

This module provides a unified interface for running virtual screening
pipeline stages as SLURM array jobs.

Usage:
    # Orchestrator mode (submits array jobs):
    python -m workflow.slurm.run --stage docking

    # Worker mode (called by SLURM array tasks):
    python -m workflow.slurm.run --stage docking --worker --chunk-id $SLURM_ARRAY_TASK_ID

    # Devel mode (small subset, devel partition defaults):
    python -m workflow.slurm.run --stage docking --devel
"""

from .manifest import query_pending, update_completed
from .jobs import create_chunks, submit_array, wait_for_job, collect_results

__all__ = [
    'query_pending',
    'update_completed',
    'create_chunks',
    'submit_array',
    'wait_for_job',
    'collect_results',
]
