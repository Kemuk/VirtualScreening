#!/usr/bin/env python3
"""
submit_stage.py

Unified SLURM array job submitter for all pipeline stages.

Replaces individual submit_*_array.sh scripts with a single Python interface.
Integrates with existing infrastructure:
  - prepare_stage.py: Filters manifest and creates chunks
  - stage_config.py: Provides resource specifications
  - {stage}.slurm: Stage-specific SLURM templates
  - update_manifest.py: Merges results back to manifest

Usage:
    # Submit single stage
    python -m workflow.slurm.submit_stage --stage docking

    # Submit with devel mode
    python -m workflow.slurm.submit_stage --stage docking --mode devel

    # Submit docking with CPU mode
    python -m workflow.slurm.submit_stage --stage docking --docking-mode cpu

    # Submit multiple stages
    python -m workflow.slurm.submit_stage --stage ligands,docking,conversion

    # Just prepare, don't submit
    python -m workflow.slurm.submit_stage --stage docking --prepare-only
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from workflow.slurm.stage_config import (
    get_stage_config,
    get_stage_resources,
    list_stages,
)


def prepare_stage(
    stage: str,
    num_chunks: int,
    manifest_path: Path,
    output_dir: Path,
    max_items: Optional[int] = None,
) -> Optional[int]:
    """
    Prepare stage by filtering manifest and creating chunks.

    Args:
        stage: Stage name
        num_chunks: Number of chunks for array job
        manifest_path: Path to manifest.parquet
        output_dir: Output directory for pending parquet
        max_items: Maximum items to process (for devel mode)

    Returns:
        Number of actual chunks created, or None if nothing to do
    """
    # Build command
    cmd = [
        sys.executable,
        "-m",
        "workflow.slurm.prepare_stage",
        "--stage",
        stage,
        "--num-chunks",
        str(num_chunks),
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(output_dir),
    ]

    if max_items is not None:
        cmd.extend(["--max-items", str(max_items)])

    print(f"Preparing stage: {stage}")
    print(f"  Command: {' '.join(cmd)}")

    # Run prepare_stage.py
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    # Print output
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Check for errors
    if result.returncode != 0:
        print(f"ERROR: prepare_stage.py failed with code {result.returncode}")
        sys.exit(1)

    # Check if nothing to do
    if "Nothing to do" in result.stdout:
        print(f"  → Nothing to do for {stage} (already complete)")
        return None

    # Extract actual chunk count
    for line in result.stdout.split("\n"):
        if "Actual chunks:" in line:
            try:
                actual_chunks = int(line.split(":")[1].strip().split()[0])
                print(f"  → Created {actual_chunks} chunks")
                return actual_chunks
            except (IndexError, ValueError):
                pass

    print("ERROR: Could not determine chunk count from prepare_stage.py output")
    sys.exit(1)


def submit_array_job(
    stage: str,
    actual_chunks: int,
    cluster: str,
    resources: dict,
    project_dir: Path,
    log_dir: Path,
    dependency: Optional[str] = None,
) -> str:
    """
    Submit SLURM array job for a stage.

    Args:
        stage: Stage name
        actual_chunks: Number of chunks to process
        cluster: SLURM cluster name
        resources: Resource dict (time, mem, cpus, partition, etc.)
        project_dir: Project root directory
        log_dir: Directory for SLURM logs
        dependency: Optional job dependency (afterok:jobid)

    Returns:
        Job ID of submitted array job
    """
    # Build sbatch command
    array_end = actual_chunks - 1
    max_concurrent = resources.get("max_concurrent", 100)

    sbatch_cmd = [
        "sbatch",
        "--parsable",
        f"--array=0-{array_end}%{max_concurrent}",
        f"--output={log_dir}/{stage}_%A_%a.out",
        f"--error={log_dir}/{stage}_%A_%a.err",
        f"--export=ALL,PROJECT_DIR={project_dir},NUM_CHUNKS={actual_chunks}",
        f"--clusters={cluster}",
        f"--time={resources['time']}",
        f"--mem={resources['mem']}",
        f"--cpus-per-task={resources['cpus']}",
    ]

    # Add partition if specified
    if "partition" in resources:
        sbatch_cmd.append(f"--partition={resources['partition']}")

    # Add GPUs if specified
    if "gpus" in resources:
        sbatch_cmd.append(f"--gpus={resources['gpus']}")

    # Add dependency if specified
    if dependency:
        sbatch_cmd.append(f"--dependency={dependency}")

    # Add SLURM template
    config = get_stage_config(stage)
    slurm_template = project_dir / "workflow" / "slurm" / config["slurm_template"]
    sbatch_cmd.append(str(slurm_template))

    print(f"\nSubmitting array job for {stage}:")
    print(f"  Command: {' '.join(sbatch_cmd)}")

    # Submit job
    result = subprocess.run(
        sbatch_cmd,
        capture_output=True,
        text=True,
        cwd=project_dir,
    )

    if result.returncode != 0:
        print(f"ERROR: sbatch failed with code {result.returncode}")
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    # Extract job ID (format: "JOBID;CLUSTER" or just "JOBID")
    job_id_raw = result.stdout.strip()
    job_id = job_id_raw.split(";")[0]

    print(f"  → Array job ID: {job_id}")
    return job_id


def submit_update_job(
    stage: str,
    array_job_id: str,
    cluster: str,
    resources: dict,
    project_dir: Path,
    log_dir: Path,
    mode: str,
) -> str:
    """
    Submit update_manifest job that runs after array job completes.

    Args:
        stage: Stage name
        array_job_id: Job ID of array job to depend on
        cluster: SLURM cluster name
        resources: Resource dict (use same cluster as array job)
        project_dir: Project root directory
        log_dir: Directory for SLURM logs

    Returns:
        Job ID of submitted update job
    """
    # Use shorter time for update job
    if mode=="devel":
        update_time = "00:10:00"
        update_partition = resources.get("partition", "devel")

    else:
        update_time = "00:30:00"
        update_partition = resources.get("partition", "short")

    sbatch_cmd = [
        "sbatch",
        "--parsable",
        f"--dependency=afterok:{array_job_id}",
        f"--output={log_dir}/update_{stage}_%j.out",
        f"--error={log_dir}/update_{stage}_%j.err",
        f"--export=ALL,PROJECT_DIR={project_dir},STAGE={stage}",
        f"--clusters={cluster}",
        f"--partition={update_partition}",
        f"--time={update_time}",
        str(project_dir / "workflow" / "slurm" / "update_manifest.slurm"),
    ]

    print(f"\nSubmitting update job for {stage}:")
    print(f"  Command: {' '.join(sbatch_cmd)}")

    # Submit job
    result = subprocess.run(
        sbatch_cmd,
        capture_output=True,
        text=True,
        cwd=project_dir,
    )

    if result.returncode != 0:
        print(f"ERROR: sbatch for update job failed with code {result.returncode}")
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    # Extract job ID
    job_id_raw = result.stdout.strip()
    job_id = job_id_raw.split(";")[0]

    print(f"  → Update job ID: {job_id} (depends on {array_job_id})")
    return job_id


def submit_stage_pipeline(
    stage: str,
    mode: str = "production",
    docking_mode: str = "gpu",
    num_chunks: Optional[int] = None,
    max_items: Optional[int] = None,
    manifest_path: Optional[Path] = None,
    project_dir: Optional[Path] = None,
    prepare_only: bool = False,
    dependency: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Submit complete pipeline for a single stage.

    Args:
        stage: Stage name
        mode: 'production' or 'devel'
        docking_mode: 'gpu' or 'cpu' (only for docking stage)
        num_chunks: Number of chunks (defaults from config if not specified)
        max_items: Max items to process (devel mode default: 1000)
        manifest_path: Path to manifest (default: data/master/manifest.parquet)
        project_dir: Project root (default: current directory)
        prepare_only: If True, only prepare stage, don't submit jobs
        dependency: Optional job dependency (afterok:jobid)

    Returns:
        Tuple of (array_job_id, update_job_id) or (None, None) if nothing to do
    """
    # Set defaults
    if project_dir is None:
        project_dir = Path.cwd()
    if manifest_path is None:
        manifest_path = project_dir / "data" / "master" / "manifest.parquet"

    # Get stage configuration
    config = get_stage_config(stage)
    resources = get_stage_resources(stage, mode, docking_mode)

    # Set num_chunks from config if not specified
    if num_chunks is None:
        if mode == "devel":
            num_chunks = 5
        else:
            num_chunks = 500  # Conservative default

    # Set max_items for devel mode
    if mode == "devel" and max_items is None:
        max_items = 1000

    print("=" * 60)
    print(f"STAGE: {stage}")
    print(f"  Description: {config['description']}")
    print(f"  Mode: {mode}")
    if stage == "docking":
        print(f"  Docking mode: {docking_mode}")
    print(f"  Chunks: {num_chunks}")
    if max_items:
        print(f"  Max items: {max_items}")
    print(f"  Cluster: {config['cluster']}")
    print("=" * 60)

    # Step 1: Prepare stage (filter manifest, create chunks)
    output_dir = project_dir / "data" / "master"
    actual_chunks = prepare_stage(
        stage=stage,
        num_chunks=num_chunks,
        manifest_path=manifest_path,
        output_dir=output_dir,
        max_items=max_items,
    )

    # If nothing to do or prepare-only, return early
    if actual_chunks is None:
        return (None, None)

    if prepare_only:
        print(f"\n✓ Prepared {stage} (--prepare-only, not submitting jobs)")
        return (None, None)

    # Step 2: Submit array job
    log_dir = project_dir / "data" / "logs" / "slurm"
    log_dir.mkdir(parents=True, exist_ok=True)

    array_job_id = submit_array_job(
        stage=stage,
        actual_chunks=actual_chunks,
        cluster=config["cluster"],
        resources=resources,
        project_dir=project_dir,
        log_dir=log_dir,
        dependency=dependency,
    )

    # Step 3: Submit update manifest job
    update_job_id = submit_update_job(
        stage=stage,
        array_job_id=array_job_id,
        cluster=config["cluster"],
        resources=resources,
        project_dir=project_dir,
        log_dir=log_dir,
        mode=mode,
    )

    print(f"\n✓ Submitted {stage}")
    print(f"  Array job: {array_job_id}")
    print(f"  Update job: {update_job_id}\n")

    return (array_job_id, update_job_id)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified SLURM array job submitter for pipeline stages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        help=f"Stage(s) to submit (comma-separated). Valid: {', '.join(list_stages())}",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["production", "devel"],
        default="production",
        help="Execution mode (default: production)",
    )

    parser.add_argument(
        "--docking-mode",
        type=str,
        choices=["gpu", "cpu"],
        default="gpu",
        help="Docking mode for docking stage (default: gpu)",
    )

    parser.add_argument(
        "--num-chunks",
        type=int,
        help="Number of chunks (default: from config or 500 prod / 5 devel)",
    )

    parser.add_argument(
        "--max-items",
        type=int,
        help="Max items to process (default: unlimited prod / 1000 devel)",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to manifest (default: data/master/manifest.parquet)",
    )

    parser.add_argument(
        "--project-dir",
        type=Path,
        help="Project root directory (default: current directory)",
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare stage, don't submit jobs",
    )

    args = parser.parse_args()

    # Split stages by comma
    stages = [s.strip() for s in args.stage.split(",")]

    # Validate stages
    valid_stages = list_stages()
    for stage in stages:
        if stage not in valid_stages:
            print(f"ERROR: Invalid stage '{stage}'", file=sys.stderr)
            print(f"Valid stages: {', '.join(valid_stages)}", file=sys.stderr)
            sys.exit(1)

    # Track submitted jobs
    submitted_jobs = []
    last_update_job_id = None

    # Submit each stage
    for stage in stages:
        # Chain stages with dependency on previous update job
        dependency = f"afterok:{last_update_job_id}" if last_update_job_id else None

        array_job_id, update_job_id = submit_stage_pipeline(
            stage=stage,
            mode=args.mode,
            docking_mode=args.docking_mode,
            num_chunks=args.num_chunks,
            max_items=args.max_items,
            manifest_path=args.manifest,
            project_dir=args.project_dir,
            prepare_only=args.prepare_only,
            dependency=dependency,
        )

        if array_job_id and update_job_id:
            submitted_jobs.append((stage, array_job_id, update_job_id))
            last_update_job_id = update_job_id

    # Print summary
    if submitted_jobs:
        print("=" * 60)
        print("ALL JOBS SUBMITTED")
        print("=" * 60)
        print("\nSubmitted jobs:")
        for stage, array_id, update_id in submitted_jobs:
            print(f"  {stage:15s} array={array_id:10s} update={update_id}")

        print("\nMonitor with:")
        print("  squeue -u $USER")
        print("  sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed")
        print("\nLogs in: data/logs/slurm/")
        print("=" * 60)
    else:
        print("\n✓ No jobs submitted (all stages complete or prepare-only mode)")

    sys.exit(0)


if __name__ == "__main__":
    main()
