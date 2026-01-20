"""
workers - Stage-specific worker modules for SLURM array jobs.

Each worker module provides a process_slice() function that:
1. Reads its slice from the pending parquet
2. Processes each item
3. Writes results to CSV
"""

from pathlib import Path
from math import ceil
from typing import List, Optional

import pandas as pd
import pyarrow.parquet as pq


# Define expected columns for each stage's results CSV
# This ensures consistent output and helps with debugging
STAGE_COLUMNS = {
    'ligands': [
        'compound_key',
        'ligand_id',
        'smiles',
        'ligand_pdbqt_path',
        'success',
        'skipped',
        'error',
    ],
    'docking': [
        'compound_key',
        'ligand_id',
        'ligand_pdbqt_path',
        'docked_pdbqt_path',
        'success',
        'skipped',
        'score',
        'error',
    ],
    'conversion': [
        'compound_key',
        'ligand_id',
        'docked_pdbqt_path',
        'docked_sdf_path',
        'success',
        'skipped',
        'error',
    ],
    'aev_infer': [
        'compound_key',
        'ligand_id',
        'docked_sdf_path',
        'success',
        'score',
        'error',
    ],
}


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
    columns: Optional[List[str]] = None,
) -> Path:
    """
    Write worker results to CSV with consistent columns.

    Args:
        results: List of result dicts
        results_dir: Directory for result files
        stage: Stage name
        task_id: SLURM_ARRAY_TASK_ID
        columns: Optional list of columns to include (uses STAGE_COLUMNS if not provided)

    Returns:
        Path to written CSV file
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{stage}_{task_id:05d}.csv"

    # Get expected columns for this stage
    if columns is None:
        columns = STAGE_COLUMNS.get(stage, None)

    df = pd.DataFrame(results)

    # Ensure all expected columns exist (fill missing with empty string)
    if columns:
        for col in columns:
            if col not in df.columns:
                df[col] = ''
        # Reorder to match expected column order
        df = df[columns]

    df.to_csv(output_path, index=False)

    return output_path
