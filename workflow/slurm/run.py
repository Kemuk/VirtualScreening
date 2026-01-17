#!/usr/bin/env python3
"""
run.py

Unified entry point for SLURM array job orchestration.

Two modes of operation:
  1. Orchestrator mode: Query manifest, chunk items, submit array job, wait, collect results
  2. Worker mode: Read chunk, process items, write results

Usage:
    # Orchestrator mode
    python -m workflow.slurm.run --stage docking

    # Worker mode (called by SLURM)
    python -m workflow.slurm.run --stage docking --worker --chunk-id 0

    # Devel mode (15 items, 1 minute timeout)
    python -m workflow.slurm.run --stage docking --devel
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Optional

from .manifest import query_pending, update_completed, get_stage_progress, STAGE_CONFIG
from .jobs import (
    create_chunks, read_chunk, submit_array,
    wait_for_job, collect_results, write_results,
    cleanup_stage_files, PARTITION_CONFIG,
)


# Stage configuration: maps stage name to processing function and resources
STAGES = {
    'manifest': {
        'function': 'workflow.scripts.create_manifest.process_batch',
        'partition': 'arc',
        'time': 10,
        'mem': '8G',
    },
    'receptors': {
        'function': 'workflow.scripts.mol2_to_pdbqt.process_batch',
        'partition': 'arc',
        'time': 5,
        'mem': '4G',
    },
    'ligands': {
        'function': 'workflow.scripts.prepare_all_ligands.process_batch',
        'partition': 'arc',
        'time': 30,
        'mem': '8G',
    },
    'docking': {
        'function': 'workflow.scripts.dock_vina.process_batch',
        'partition': 'arc',
        'time': 60,
        'mem': '16G',
    },
    'conversion': {
        'function': 'workflow.scripts.pdbqt_to_sdf.process_batch',
        'partition': 'arc',
        'time': 10,
        'mem': '4G',
    },
    'aev_prep': {
        'function': 'workflow.scripts.prepare_aev_plig_csv.process_batch',
        'partition': 'arc',
        'time': 20,
        'mem': '8G',
    },
    'aev_infer': {
        'function': 'workflow.scripts.rescore_aev_plig.process_batch',
        'partition': 'htc',
        'time': 120,
        'mem': '16G',
        'gres': 'gpu:1',
    },
    'aev_merge': {
        'function': 'workflow.scripts.update_manifest_aev_plig.process_batch',
        'partition': 'arc',
        'time': 10,
        'mem': '8G',
    },
    'results': {
        'function': 'workflow.scripts.compute_results.process_batch',
        'partition': 'arc',
        'time': 30,
        'mem': '16G',
    },
}

# Devel mode overrides
DEVEL_CONFIG = {
    'max_items': 15,
    'time': 1,  # 1 minute
    'partition': 'devel',
}


def load_config(config_path: Path) -> dict:
    """Load workflow configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def import_function(function_path: str):
    """Dynamically import a function from a module path."""
    module_path, function_name = function_path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[function_name])
    return getattr(module, function_name)


def run_orchestrator(
    stage: str,
    config_path: Path,
    devel: bool = False,
    max_items: Optional[int] = None,
    time_limit: Optional[int] = None,
) -> bool:
    """
    Run orchestrator mode: submit and manage SLURM array job.

    Args:
        stage: Stage name
        config_path: Path to config.yaml
        devel: Use devel mode settings
        max_items: Override max items
        time_limit: Override time limit (minutes)

    Returns:
        True if successful
    """
    if stage not in STAGES:
        print(f"ERROR: Unknown stage '{stage}'", file=sys.stderr)
        print(f"Valid stages: {list(STAGES.keys())}", file=sys.stderr)
        return False

    config = load_config(config_path)
    stage_config = STAGES[stage].copy()

    # Apply devel overrides
    if devel:
        max_items = max_items or DEVEL_CONFIG['max_items']
        time_limit = time_limit or DEVEL_CONFIG['time']
        stage_config['partition'] = DEVEL_CONFIG['partition']

    if time_limit:
        stage_config['time'] = time_limit

    # Paths
    project_root = config_path.parent.parent
    manifest_path = project_root / config['manifest_dir'] / 'manifest.parquet'
    chunk_dir = project_root / 'data' / 'slurm' / 'chunks'
    results_dir = project_root / 'data' / 'slurm' / 'results'
    logs_dir = project_root / 'data' / 'slurm' / 'logs'
    script_path = project_root / 'workflow' / 'slurm' / 'array_job.sh'

    # Show progress
    print(f"\n{'='*60}")
    print(f"Stage: {stage}")
    print(f"{'='*60}")

    progress = get_stage_progress(manifest_path, stage)
    print(f"Progress: {progress['completed']}/{progress['total']} ({progress['percent']:.1f}%)")
    print(f"Pending: {progress['pending']} items")

    if progress['pending'] == 0:
        print("Nothing to do - stage complete!")
        return True

    # Query pending items
    print(f"\nQuerying manifest...")
    items = query_pending(manifest_path, stage, max_items=max_items)
    print(f"Found {len(items)} items to process")

    if len(items) == 0:
        print("No items to process")
        return True

    # Create chunks
    partition = stage_config['partition']
    print(f"\nCreating chunks (partition={partition})...")
    n_chunks = create_chunks(items, chunk_dir, stage, partition)
    print(f"Created {n_chunks} chunks")

    # Submit array job
    print(f"\nSubmitting SLURM array job...")
    job_id = submit_array(
        stage=stage,
        n_chunks=n_chunks,
        partition=partition,
        time_minutes=stage_config['time'],
        mem=stage_config['mem'],
        script_path=script_path,
        chunk_dir=chunk_dir,
        results_dir=results_dir,
        logs_dir=logs_dir,
        gres=stage_config.get('gres'),
        config_path=config_path,
    )

    # Wait for completion
    print(f"\nWaiting for job {job_id}...")
    summary = wait_for_job(job_id)

    print(f"\nJob summary:")
    print(f"  Total tasks: {summary['total']}")
    print(f"  Completed: {summary['completed']}")
    print(f"  Failed: {summary['failed']}")

    if not summary['success']:
        print(f"\nERROR: {summary['failed']} tasks failed", file=sys.stderr)
        return False

    # Collect results
    print(f"\nCollecting results...")
    results = collect_results(results_dir, stage)
    print(f"Collected {len(results)} results")

    # Update manifest
    completed_ids = [r['ligand_id'] for r in results if r.get('success', True)]
    scores = {r['ligand_id']: r.get('score') for r in results if 'score' in r}

    print(f"\nUpdating manifest...")
    n_updated = update_completed(manifest_path, stage, completed_ids, scores)
    print(f"Updated {n_updated} items")

    # Cleanup
    cleanup_stage_files(chunk_dir, results_dir, stage)

    print(f"\n{'='*60}")
    print(f"Stage {stage} complete!")
    print(f"{'='*60}\n")

    return True


def run_worker(
    stage: str,
    chunk_id: int,
    chunk_dir: Path,
    results_dir: Path,
    config_path: Optional[Path] = None,
) -> bool:
    """
    Run worker mode: process a single chunk.

    Args:
        stage: Stage name
        chunk_id: Chunk index (from SLURM_ARRAY_TASK_ID)
        chunk_dir: Directory containing chunk files
        results_dir: Directory for result files
        config_path: Path to config.yaml

    Returns:
        True if successful
    """
    if stage not in STAGES:
        print(f"ERROR: Unknown stage '{stage}'", file=sys.stderr)
        return False

    stage_config = STAGES[stage]

    # Load config if provided
    config = load_config(config_path) if config_path else {}

    # Read chunk
    print(f"Worker: stage={stage}, chunk_id={chunk_id}")
    items = read_chunk(chunk_dir, stage, chunk_id)
    print(f"Processing {len(items)} items")

    # Import and call processing function
    try:
        process_func = import_function(stage_config['function'])
    except (ImportError, AttributeError) as e:
        print(f"ERROR: Cannot import {stage_config['function']}: {e}", file=sys.stderr)
        return False

    # Process items
    try:
        results = process_func(items, config)
    except Exception as e:
        print(f"ERROR: Processing failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

    # Write results
    write_results(results_dir, stage, chunk_id, results)
    print(f"Wrote {len(results)} results")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="SLURM Array Job Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run docking stage
  python -m workflow.slurm.run --stage docking

  # Devel mode (15 items, 1 minute)
  python -m workflow.slurm.run --stage docking --devel

  # Custom limits
  python -m workflow.slurm.run --stage docking --max-items 100 --time 5

  # Worker mode (called by SLURM)
  python -m workflow.slurm.run --stage docking --worker --chunk-id 0
""",
    )

    parser.add_argument(
        '--stage',
        type=str,
        required=True,
        choices=list(STAGES.keys()),
        help='Pipeline stage to run',
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config/config.yaml'),
        help='Path to config.yaml',
    )
    parser.add_argument(
        '--devel',
        action='store_true',
        help='Use devel mode (15 items, 1 minute, devel partition)',
    )
    parser.add_argument(
        '--max-items',
        type=int,
        help='Maximum items to process',
    )
    parser.add_argument(
        '--time',
        type=int,
        help='Time limit in minutes',
    )

    # Worker mode arguments
    parser.add_argument(
        '--worker',
        action='store_true',
        help='Run in worker mode (called by SLURM array task)',
    )
    parser.add_argument(
        '--chunk-id',
        type=int,
        help='Chunk ID (SLURM_ARRAY_TASK_ID)',
    )
    parser.add_argument(
        '--chunk-dir',
        type=Path,
        help='Chunk directory (worker mode)',
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        help='Results directory (worker mode)',
    )

    args = parser.parse_args()

    # Validate arguments
    if args.worker:
        if args.chunk_id is None:
            parser.error("--chunk-id required in worker mode")
        if args.chunk_dir is None:
            parser.error("--chunk-dir required in worker mode")
        if args.results_dir is None:
            parser.error("--results-dir required in worker mode")

        success = run_worker(
            stage=args.stage,
            chunk_id=args.chunk_id,
            chunk_dir=args.chunk_dir,
            results_dir=args.results_dir,
            config_path=args.config,
        )
    else:
        success = run_orchestrator(
            stage=args.stage,
            config_path=args.config,
            devel=args.devel,
            max_items=args.max_items,
            time_limit=args.time,
        )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
