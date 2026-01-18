"""
jobs.py

SLURM array job management: chunking, submission, waiting, and result collection.

This module handles:
  - Splitting items into chunks respecting MaxArraySize
  - Writing chunk files for workers to read
  - Submitting SLURM array jobs
  - Polling for job completion
  - Collecting results from chunk output files
"""

import json
import subprocess
import time
import re
from pathlib import Path
from typing import Optional
import pandas as pd


# SLURM queue limits (cluster or partition key)
PARTITION_CONFIG = {
    'arc': {
        'max_array_size': 5000,
        'default_time': 60,      # minutes
        'default_mem': '8G',
    },
    'htc': {
        'max_array_size': 1000,
        'default_time': 120,
        'default_mem': '16G',
        'gres': 'gpu:1',
    },
    'devel': {
        'max_array_size': 100,
        'default_time': 1,       # 1 minute for devel
        'default_mem': '4G',
    },
}


def create_chunks(
    items: pd.DataFrame,
    chunk_dir: Path,
    stage: str,
    partition: str = 'arc',
) -> int:
    """
    Split items into chunks and write chunk files.

    Each chunk file contains a JSON list of item records that
    the worker will process.

    Args:
        items: DataFrame of items to process
        chunk_dir: Directory to write chunk files
        stage: Stage name (for subdirectory)
        partition: Queue profile key (determines max array size)

    Returns:
        Number of chunks created
    """
    max_array = PARTITION_CONFIG.get(partition, {}).get('max_array_size', 1000)

    # Calculate chunk size to stay within array limits
    n_items = len(items)
    if n_items == 0:
        return 0

    n_chunks = min(n_items, max_array)
    chunk_size = (n_items + n_chunks - 1) // n_chunks  # Ceiling division

    # Create chunk directory
    stage_chunk_dir = chunk_dir / stage
    stage_chunk_dir.mkdir(parents=True, exist_ok=True)

    # Clear old chunk files
    for old_file in stage_chunk_dir.glob('chunk_*.json'):
        old_file.unlink()

    # Write chunk files
    records = items.to_dict('records')
    chunk_id = 0

    for i in range(0, n_items, chunk_size):
        chunk_records = records[i:i + chunk_size]
        chunk_file = stage_chunk_dir / f'chunk_{chunk_id:05d}.json'

        with open(chunk_file, 'w') as f:
            json.dump(chunk_records, f, indent=2, default=str)

        chunk_id += 1

    return chunk_id


def read_chunk(chunk_dir: Path, stage: str, chunk_id: int) -> list:
    """
    Read a chunk file.

    Args:
        chunk_dir: Base chunk directory
        stage: Stage name
        chunk_id: Chunk index (from SLURM_ARRAY_TASK_ID)

    Returns:
        List of item records
    """
    chunk_file = chunk_dir / stage / f'chunk_{chunk_id:05d}.json'

    if not chunk_file.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunk_file}")

    with open(chunk_file) as f:
        return json.load(f)


def submit_array(
    stage: str,
    n_chunks: int,
    cluster: Optional[str],
    partition: Optional[str],
    time_minutes: int,
    mem: str,
    script_path: Path,
    chunk_dir: Path,
    results_dir: Path,
    logs_dir: Path,
    gres: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> str:
    """
    Submit a SLURM array job.

    Args:
        stage: Stage name
        n_chunks: Number of array tasks
        cluster: SLURM cluster name (e.g., arc, htc)
        partition: SLURM partition (e.g., devel)
        time_minutes: Time limit in minutes
        mem: Memory allocation (e.g., '8G')
        script_path: Path to array_job.sh
        chunk_dir: Directory containing chunk files
        results_dir: Directory for result files
        logs_dir: Directory for SLURM logs
        gres: GPU resources (e.g., 'gpu:1')
        config_path: Path to config.yaml

    Returns:
        SLURM job ID
    """
    script_path = script_path.resolve()
    chunk_dir = chunk_dir.resolve()
    results_dir = results_dir.resolve()
    logs_dir = logs_dir.resolve()
    if config_path:
        config_path = config_path.resolve()

    # Create log directory
    stage_logs = logs_dir / stage
    stage_logs.mkdir(parents=True, exist_ok=True)

    # Create results directory
    stage_results = results_dir / stage
    stage_results.mkdir(parents=True, exist_ok=True)

    # Format time as HH:MM:SS
    hours = time_minutes // 60
    mins = time_minutes % 60
    time_str = f"{hours:02d}:{mins:02d}:00"

    # Build sbatch command
    cmd = [
        'sbatch',
        f'--array=0-{n_chunks - 1}',
        f'--time={time_str}',
        f'--mem={mem}',
        f'--job-name=vs-{stage}',
        f'--output={stage_logs}/slurm_%A_%a.out',
        f'--error={stage_logs}/slurm_%A_%a.err',
    ]

    if cluster:
        cmd.append(f'--clusters={cluster}')

    if partition:
        cmd.append(f'--partition={partition}')

    if gres:
        cmd.append(f'--gres={gres}')

    # Add script and arguments
    cmd.extend([
        str(script_path),
        stage,
        str(chunk_dir),
        str(stage_results),
    ])

    if config_path:
        cmd.append(str(config_path))

    # Submit job
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    # Parse job ID from output: "Submitted batch job 12345"
    match = re.search(r'Submitted batch job (\d+)', result.stdout)
    if match:
        job_id = match.group(1)
        print(f"Submitted SLURM array job: {job_id} ({n_chunks} tasks)")
        return job_id

    raise RuntimeError(f"Failed to parse job ID from: {result.stdout}")


def wait_for_job(
    job_id: str,
    poll_interval: int = 10,
    timeout: Optional[int] = None,
) -> dict:
    """
    Wait for a SLURM job to complete.

    Args:
        job_id: SLURM job ID
        poll_interval: Seconds between status checks
        timeout: Maximum wait time in seconds (None = no limit)

    Returns:
        Dict with job status summary
    """
    start_time = time.time()
    last_status = None

    while True:
        # Check if job still exists in queue
        result = subprocess.run(
            ['squeue', '-j', job_id, '-h', '-o', '%T'],
            capture_output=True,
            text=True,
        )

        statuses = result.stdout.strip().split('\n')
        statuses = [s for s in statuses if s]  # Remove empty strings

        if not statuses:
            # Job no longer in queue - check final status with sacct
            return get_job_summary(job_id)

        current_status = statuses[0] if len(set(statuses)) == 1 else 'MIXED'

        if current_status != last_status:
            print(f"Job {job_id}: {current_status} ({len(statuses)} tasks)")
            last_status = current_status

        # Check timeout
        if timeout and (time.time() - start_time) > timeout:
            raise TimeoutError(f"Job {job_id} exceeded timeout of {timeout}s")

        time.sleep(poll_interval)


def get_job_summary(job_id: str) -> dict:
    """
    Get summary of completed job using sacct.

    Args:
        job_id: SLURM job ID

    Returns:
        Dict with 'completed', 'failed', 'total' counts
    """
    result = subprocess.run(
        [
            'sacct', '-j', job_id,
            '--format=JobID,State,ExitCode',
            '--noheader', '--parsable2',
        ],
        capture_output=True,
        text=True,
    )

    completed = 0
    failed = 0
    total = 0

    for line in result.stdout.strip().split('\n'):
        if not line or '.batch' in line or '.extern' in line:
            continue

        parts = line.split('|')
        if len(parts) >= 2:
            job_part, state = parts[0], parts[1]
            # Only count array tasks (format: jobid_taskid)
            if '_' in job_part:
                total += 1
                if state == 'COMPLETED':
                    completed += 1
                elif state in ('FAILED', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL'):
                    failed += 1

    return {
        'job_id': job_id,
        'total': total,
        'completed': completed,
        'failed': failed,
        'success': failed == 0,
    }


def collect_results(results_dir: Path, stage: str) -> list:
    """
    Collect results from all chunk output files.

    Each worker writes a JSON file with results for its chunk.
    This function aggregates all chunk results.

    Args:
        results_dir: Base results directory
        stage: Stage name

    Returns:
        List of all result records
    """
    stage_results = results_dir / stage
    all_results = []

    for result_file in sorted(stage_results.glob('result_*.json')):
        with open(result_file) as f:
            chunk_results = json.load(f)
            all_results.extend(chunk_results)

    return all_results


def write_results(
    results_dir: Path,
    stage: str,
    chunk_id: int,
    results: list,
) -> None:
    """
    Write results for a chunk.

    Called by workers after processing their chunk.

    Args:
        results_dir: Base results directory
        stage: Stage name
        chunk_id: Chunk index
        results: List of result records
    """
    stage_results = results_dir / stage
    stage_results.mkdir(parents=True, exist_ok=True)

    result_file = stage_results / f'result_{chunk_id:05d}.json'

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def cleanup_stage_files(
    chunk_dir: Path,
    results_dir: Path,
    stage: str,
) -> None:
    """
    Clean up chunk and result files after successful stage completion.

    Args:
        chunk_dir: Base chunk directory
        results_dir: Base results directory
        stage: Stage name
    """
    # Remove chunk files
    stage_chunks = chunk_dir / stage
    if stage_chunks.exists():
        for f in stage_chunks.glob('chunk_*.json'):
            f.unlink()

    # Remove result files
    stage_results = results_dir / stage
    if stage_results.exists():
        for f in stage_results.glob('result_*.json'):
            f.unlink()
