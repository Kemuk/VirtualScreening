"""
workers - Stage-specific worker modules for SLURM array jobs.

Each worker module provides a process_slice() function that:
1. Reads its slice from the pending parquet
2. Processes each item
3. Writes results to CSV
"""

from pathlib import Path
from math import ceil

import pandas as pd
import pyarrow.parquet as pq


def read_slice(
    pending_path: Path,
    task_id: int,
    num_chunks: int,
) -> pd.DataFrame:
    """
    Read a slice of the pending parquet for this worker.

    Args:
        pending_path: Path to pending parquet file
        task_id: SLURM_ARRAY_TASK_ID (0-indexed)
        num_chunks: Total number of chunks

    Returns:
        DataFrame slice for this worker
    """
    # Read full table (parquet is columnar, this is efficient)
    table = pq.read_table(pending_path)
    total_rows = table.num_rows

    # Calculate slice bounds
    chunk_size = ceil(total_rows / num_chunks)
    start = task_id * chunk_size
    end = min(start + chunk_size, total_rows)

    # Slice and convert to pandas
    if start >= total_rows:
        return pd.DataFrame()  # Empty slice for overflow tasks

    sliced = table.slice(start, end - start)
    return sliced.to_pandas()


def write_results(
    results: list,
    results_dir: Path,
    stage: str,
    task_id: int,
) -> Path:
    """
    Write worker results to CSV.

    Args:
        results: List of result dicts with compound_key, success, score, error
        results_dir: Directory for result files
        stage: Stage name
        task_id: SLURM_ARRAY_TASK_ID

    Returns:
        Path to written CSV file
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{stage}_{task_id:05d}.csv"

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    return output_path
