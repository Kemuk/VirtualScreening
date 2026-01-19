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
        'time': 15,  # 15 min per chunk (RDKit canonicalization + file checks)
        'mem': '4G',  # Lower memory per chunk since items are distributed
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
        'mem': '20G',
        'cpus': 8,
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
    'max_items': 1000,  # 1000 ligands for quick testing
    'time': 10,  # 10 minutes per task
    'partition': 'devel',
}

# Stage execution order for 'all' mode
STAGE_ORDER = [
    'manifest',
    'receptors',
    'ligands',
    'docking',
    'conversion',
    'aev_prep',
    'aev_infer',
    'aev_merge',
    'results',
]


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
    overwrite: bool = False,
) -> bool:
    """
    Run orchestrator mode: submit and manage SLURM array job.

    Args:
        stage: Stage name
        config_path: Path to config.yaml
        devel: Use devel mode settings
        max_items: Override max items
        time_limit: Override time limit (minutes)
        overwrite: Overwrite existing manifest if it exists

    Returns:
        True if successful
    """
    if stage not in STAGES:
        print(f"ERROR: Unknown stage '{stage}'", file=sys.stderr)
        print(f"Valid stages: {list(STAGES.keys())}", file=sys.stderr)
        return False

    config_path = config_path.resolve()
    config = load_config(config_path)
    stage_config = STAGES[stage].copy()

    # Apply config-driven cluster/GPU selection for docking stage
    if stage == 'docking':
        docking_mode = config.get('docking', {}).get('mode', 'cpu')
        if docking_mode == 'gpu':
            stage_config['partition'] = 'htc'
            stage_config['gres'] = 'gpu:1'
            print("Docking mode: GPU (cluster=htc, gres=gpu:1)")
        else:
            stage_config['partition'] = 'arc'
            stage_config.pop('gres', None)  # Remove gres if present
            print("Docking mode: CPU (cluster=arc)")

    # Apply devel overrides
    cluster = stage_config['partition']
    partition = None
    if devel:
        max_items = max_items or DEVEL_CONFIG['max_items']
        time_limit = time_limit or DEVEL_CONFIG['time']
        partition = DEVEL_CONFIG['partition']

    if time_limit:
        stage_config['time'] = time_limit

    # Paths
    project_root = config_path.parent.parent.resolve()
    manifest_path = (project_root / config['manifest_dir'] / 'manifest.parquet').resolve()
    chunk_dir = (project_root / 'data' / 'slurm' / 'chunks').resolve()
    results_dir = (project_root / 'data' / 'slurm' / 'results').resolve()
    logs_dir = (project_root / 'data' / 'slurm' / 'logs').resolve()
    script_path = (project_root / 'workflow' / 'slurm' / 'array_job.sh').resolve()

    # Show progress
    print(f"\n{'='*60}")
    print(f"Stage: {stage}")
    print(f"{'='*60}")

    # Special handling for manifest stage - uses array jobs to create manifest
    if stage == 'manifest':
        if manifest_path.exists():
            if overwrite:
                # Backup existing manifest before overwriting
                backup_dir = manifest_path.parent / "backup"
                backup_dir.mkdir(parents=True, exist_ok=True)
                from datetime import datetime
                import shutil
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = backup_dir / f"manifest_{timestamp}.parquet"
                shutil.copy2(manifest_path, backup_path)
                print(f"Backed up existing manifest to: {backup_path}")
                manifest_path.unlink()
            else:
                print(f"Manifest already exists: {manifest_path}")
                print("To recreate, use --overwrite or delete the existing manifest first.")
                return True

        print("Creating manifest using array jobs...")

        # Phase 1: Scan items (lightweight, no RDKit)
        print("\nPhase 1: Scanning SMILES files...")
        from workflow.scripts.scan_manifest_items import scan_targets
        targets_path = project_root / config.get('targets_config', 'config/targets.yaml')
        targets_config = load_config(targets_path)

        items = scan_targets(
            targets_config=targets_config,
            workflow_config=config,
            project_root=project_root,
        )
        print(f"Found {len(items)} items to process")

        if len(items) == 0:
            print("ERROR: No items found. Check your SMILES files.", file=sys.stderr)
            return False

        # Convert to DataFrame for chunking
        import pandas as pd
        items_df = pd.DataFrame(items)

        # Phase 2: Create chunks and submit array job
        print("\nPhase 2: Creating chunks and submitting array job...")
        queue_profile = partition or stage_config['partition']
        partition_label = partition or "none"
        print(f"Cluster: {cluster}, partition: {partition_label}")

        n_chunks = create_chunks(items_df, chunk_dir, stage, queue_profile)
        print(f"Created {n_chunks} chunks")

        # Submit array job
        job_id = submit_array(
            stage=stage,
            n_chunks=n_chunks,
            cluster=cluster,
            partition=partition,
            time_minutes=stage_config['time'],
            mem=stage_config['mem'],
            script_path=script_path,
            chunk_dir=chunk_dir,
            results_dir=results_dir,
            logs_dir=logs_dir,
            gres=stage_config.get('gres'),
            cpus=stage_config.get('cpus'),
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

        # Phase 3: Build manifest from results
        print("\nPhase 3: Building manifest from results...")
        results = collect_results(results_dir, stage)
        print(f"Collected {len(results)} results")

        # Filter successful results and build manifest
        successful = [r for r in results if r.get('success', True)]
        failed = [r for r in results if not r.get('success', True)]

        if failed:
            print(f"Warning: {len(failed)} items failed processing")

        if not successful:
            print("ERROR: No successful results to build manifest from", file=sys.stderr)
            return False

        # Build manifest using build_manifest.py logic
        from workflow.scripts.build_manifest import build_manifest
        n_entries = build_manifest(results, manifest_path)

        # Cleanup
        cleanup_stage_files(chunk_dir, results_dir, stage)

        print(f"\n{'='*60}")
        print(f"Manifest created successfully!")
        print(f"  Total entries: {n_entries}")
        print(f"  Location: {manifest_path}")
        print(f"{'='*60}\n")
        return True

    # For all other stages, check manifest exists
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
        print("Run 'python -m workflow.slurm.run --stage manifest' first.", file=sys.stderr)
        return False

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
    queue_profile = partition or stage_config['partition']
    partition_label = partition or "none"
    print(f"\nCreating chunks (cluster={cluster}, partition={partition_label})...")
    n_chunks = create_chunks(items, chunk_dir, stage, queue_profile)
    print(f"Created {n_chunks} chunks")

    # Submit array job
    print(f"\nSubmitting SLURM array job...")
    job_id = submit_array(
        stage=stage,
        n_chunks=n_chunks,
        cluster=cluster,
        partition=partition,
        time_minutes=stage_config['time'],
        mem=stage_config['mem'],
        script_path=script_path,
        chunk_dir=chunk_dir,
        results_dir=results_dir,
        logs_dir=logs_dir,
        gres=stage_config.get('gres'),
        cpus=stage_config.get('cpus'),
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

    chunk_dir = chunk_dir.resolve()
    results_dir = results_dir.resolve()
    if config_path:
        config_path = config_path.resolve()

    # Load config if provided
    config = load_config(config_path) if config_path else {}

    # Add project_root to config (needed by manifest workers)
    if config_path:
        config['project_root'] = str(config_path.parent.parent)

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


def run_all_stages(
    config_path: Path,
    devel: bool = False,
    max_items: Optional[int] = None,
    time_limit: Optional[int] = None,
    overwrite: bool = False,
) -> bool:
    """
    Run all pipeline stages sequentially.

    Args:
        config_path: Path to config.yaml
        devel: Use devel mode settings
        max_items: Override max items
        time_limit: Override time limit (minutes)
        overwrite: Overwrite existing manifest

    Returns:
        True if all stages successful
    """
    print("\n" + "=" * 60)
    print("VIRTUAL SCREENING PIPELINE - ALL STAGES")
    print("=" * 60)
    print(f"Devel mode: {devel}")
    print(f"Stages: {' -> '.join(STAGE_ORDER)}")
    print("=" * 60 + "\n")

    for stage in STAGE_ORDER:
        print(f"\n>>> Starting stage: {stage}")
        success = run_orchestrator(
            stage=stage,
            config_path=config_path,
            devel=devel,
            max_items=max_items,
            time_limit=time_limit,
            overwrite=overwrite if stage == 'manifest' else False,
        )

        if not success:
            print(f"\nERROR: Stage '{stage}' failed. Stopping pipeline.", file=sys.stderr)
            return False

    print("\n" + "=" * 60)
    print("ALL STAGES COMPLETE!")
    print("=" * 60 + "\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="SLURM Array Job Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all stages
  python -m workflow.slurm.run --stage all

  # Run single stage
  python -m workflow.slurm.run --stage docking

  # Devel mode (10k items, quick timeout)
  python -m workflow.slurm.run --stage all --devel

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
        choices=list(STAGES.keys()) + ['all'],
        help='Pipeline stage to run (or "all" for full pipeline)',
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
        help='Use devel mode (10k items, quick timeout, devel partition)',
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
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing manifest (for manifest stage)',
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
        if args.stage == 'all':
            success = run_all_stages(
                config_path=args.config,
                devel=args.devel,
                max_items=args.max_items,
                time_limit=args.time,
                overwrite=args.overwrite,
            )
        else:
            success = run_orchestrator(
                stage=args.stage,
                config_path=args.config,
                devel=args.devel,
                max_items=args.max_items,
                time_limit=args.time,
                overwrite=args.overwrite,
            )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
