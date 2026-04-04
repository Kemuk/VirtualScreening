"""
workers - Stage-specific worker modules for SLURM array jobs.

Each worker module provides a process_slice() function that:
1. Reads its slice from the pending parquet
2. Processes each item
3. Writes results to parquet
"""

from math import ceil
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow.parquet as pq


# Define expected columns for each stage's results parquet
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
        'docking_log_path',
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
    'ligand_features': [
        'ligand_id',
        'protein_id',
        'success',
        'skipped',
        'error',
    ],
    'ligand_score': [
        'ligand_id',
        'protein_id',
        'ligand_based_score',
        'template_smiles',
        'template_source',
        'ligand_based_method',
        'success',
        'error',
    ],
}


def read_slice(
    pending_path: Path,
    task_id: int,
    num_chunks: int,
) -> pd.DataFrame:
    """
    Read a slice of the pending parquet for this worker (pandas).

    Args:
        pending_path: Path to pending parquet file
        task_id: SLURM_ARRAY_TASK_ID (0-indexed)
        num_chunks: Total number of chunks

    Returns:
        pandas DataFrame slice for this worker
    """
    table = pq.read_table(pending_path)
    total_rows = table.num_rows

    chunk_size = ceil(total_rows / num_chunks)
    start = task_id * chunk_size
    end = min(start + chunk_size, total_rows)

    if start >= total_rows:
        return pd.DataFrame()

    sliced = table.slice(start, end - start)
    return sliced.to_pandas()


def read_slice_pl(
    pending_path: Path,
    task_id: int,
    num_chunks: int,
):
    """
    Read a slice of the pending parquet for this worker (Polars).

    Args:
        pending_path: Path to pending parquet file
        task_id: SLURM_ARRAY_TASK_ID (0-indexed)
        num_chunks: Total number of chunks

    Returns:
        polars DataFrame slice for this worker
    """
    import polars as pl

    df = pl.read_parquet(pending_path)
    total = len(df)
    chunk_size = ceil(total / num_chunks)
    start = task_id * chunk_size
    if start >= total:
        return pl.DataFrame()
    return df.slice(start, chunk_size)


def write_results(
    results: list,
    results_dir: Path,
    stage: str,
    task_id: int,
    columns: Optional[List[str]] = None,
) -> Path:
    """
    Write worker results to parquet with consistent columns.

    Args:
        results: List of result dicts
        results_dir: Directory for result files
        stage: Stage name
        task_id: SLURM_ARRAY_TASK_ID
        columns: Optional list of columns to include (uses STAGE_COLUMNS if not provided)

    Returns:
        Path to written parquet file
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{stage}_{task_id:05d}.parquet"

    if columns is None:
        columns = STAGE_COLUMNS.get(stage, None)

    df = pd.DataFrame(results)

    if columns:
        for col in columns:
            if col not in df.columns:
                df[col] = ''

        pred_cols = [col for col in df.columns if 'pred' in col]
        other_cols = [col for col in df.columns if col not in columns and 'pred' not in col]
        ordered_cols = columns + pred_cols + other_cols
        df = df[[col for col in ordered_cols if col in df.columns]]

    df.to_parquet(output_path, index=False)
    return output_path
