"""
manifest.py

Manifest query and update operations for the SLURM orchestrator.

The manifest (parquet file) is the single source of truth for pipeline state.
This module provides functions to:
  - Query items pending processing for each stage
  - Update items as completed after processing
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional
import fcntl
import time


# Stage dependencies and status columns
# Note: Only these status columns exist in manifest:
#   - preparation_status (ligand prep)
#   - docking_status
#   - rescoring_status
# Other stages don't have dedicated status columns

STAGE_CONFIG = {
    'manifest': {
        'status_column': None,  # No status tracking for manifest creation
        'depends_on': None,
        'filter_column': None,
    },
    'receptors': {
        'status_column': None,  # No status column - check file existence
        'depends_on': None,
        'filter_column': None,
        'group_by': 'protein_id',  # One task per target
    },
    'ligands': {
        'status_column': 'preparation_status',
        'depends_on': None,
        'filter_column': None,
    },
    'docking': {
        'status_column': 'docking_status',
        'depends_on': 'preparation_status',
        'filter_column': None,
    },
    'conversion': {
        'status_column': 'conversion_status',
        'depends_on': 'docking_status',
        'filter_column': None,
        'check_file_column': 'docked_sdf_path',
    },
    'aev_prep': {
        'status_column': None,  # No status column
        'depends_on': 'docking_status',
        'filter_column': None,
    },
    'aev_infer': {
        'status_column': 'rescoring_status',
        'depends_on': 'docking_status',
        'filter_column': None,
    },
    'aev_merge': {
        'status_column': None,  # No status column
        'depends_on': 'rescoring_status',
        'filter_column': None,
    },
    'results': {
        'status_column': None,  # No status column
        'depends_on': 'rescoring_status',
        'filter_column': None,
        'group_by': 'protein_id',  # One task per target
    },
}


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """
    Load manifest from parquet file.

    Args:
        manifest_path: Path to manifest.parquet

    Returns:
        DataFrame with manifest data
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    table = pq.read_table(manifest_path)
    return table.to_pandas()


def save_manifest(df: pd.DataFrame, manifest_path: Path) -> None:
    """
    Save manifest to parquet file with file locking.

    Uses file locking to prevent concurrent writes from multiple
    array tasks updating the manifest simultaneously.

    Args:
        df: DataFrame to save
        manifest_path: Path to manifest.parquet
    """
    lock_path = manifest_path.with_suffix('.lock')

    # Acquire exclusive lock
    with open(lock_path, 'w') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            # Write to temporary file first, then rename (atomic)
            temp_path = manifest_path.with_suffix('.tmp')
            df.to_parquet(temp_path, index=False)
            temp_path.rename(manifest_path)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def query_pending(
    manifest_path: Path,
    stage: str,
    max_items: Optional[int] = None,
) -> pd.DataFrame:
    """
    Query manifest for items pending processing in a stage.

    Args:
        manifest_path: Path to manifest.parquet
        stage: Stage name (e.g., 'docking', 'conversion')
        max_items: Maximum items to return (for devel mode)

    Returns:
        DataFrame with pending items
    """
    if stage not in STAGE_CONFIG:
        raise ValueError(f"Unknown stage: {stage}. Valid: {list(STAGE_CONFIG.keys())}")

    config = STAGE_CONFIG[stage]
    df = load_manifest(manifest_path)

    # Filter by dependency (previous stage must be complete)
    if config['depends_on']:
        df = df[df[config['depends_on']] == True]

    # Filter by status (this stage must be incomplete)
    if config['status_column']:
        df = df[df[config['status_column']] == False]

    # For stages without a status column, optionally check file existence
    if not config['status_column'] and config.get('check_file_column'):
        check_file = config['check_file_column']

        def file_missing(path_str: str) -> bool:
            if pd.isna(path_str) or not path_str:
                return True
            return not Path(path_str).exists()

        df = df[df[check_file].apply(file_missing)]

    # Apply additional filter if specified
    if config.get('filter_column'):
        df = df[df[config['filter_column']] == True]

    # Limit items for devel mode
    if max_items is not None and len(df) > max_items:
        df = df.head(max_items)

    return df


def query_pending_grouped(
    manifest_path: Path,
    stage: str,
    max_items: Optional[int] = None,
) -> list:
    """
    Query manifest and return items grouped (e.g., by protein_id).

    Used for stages that process per-target rather than per-ligand.

    Args:
        manifest_path: Path to manifest.parquet
        stage: Stage name
        max_items: Maximum groups to return

    Returns:
        List of (group_key, group_df) tuples
    """
    config = STAGE_CONFIG[stage]
    df = query_pending(manifest_path, stage, max_items=None)

    group_by = config.get('group_by')
    if not group_by:
        raise ValueError(f"Stage {stage} does not support grouped queries")

    groups = [(key, group) for key, group in df.groupby(group_by)]

    if max_items is not None and len(groups) > max_items:
        groups = groups[:max_items]

    return groups


def update_completed(
    manifest_path: Path,
    stage: str,
    completed_ids: list,
    scores: Optional[dict] = None,
) -> int:
    """
    Update manifest to mark items as completed for a stage.

    Args:
        manifest_path: Path to manifest.parquet
        stage: Stage name
        completed_ids: List of compound_key or ligand_id values
        scores: Optional dict mapping id -> score value

    Returns:
        Number of items updated
    """
    if stage not in STAGE_CONFIG:
        raise ValueError(f"Unknown stage: {stage}")

    config = STAGE_CONFIG[stage]
    status_column = config['status_column']

    if not status_column:
        return 0  # Stage doesn't track status

    df = load_manifest(manifest_path)

    # Find matching rows
    id_column = 'ligand_id'  # Primary key column
    mask = df[id_column].isin(completed_ids)

    # Update status
    df.loc[mask, status_column] = True

    # Update scores if provided
    if scores:
        score_column = f"{stage}_score"
        if score_column not in df.columns:
            df[score_column] = None
        for item_id, score in scores.items():
            df.loc[df[id_column] == item_id, score_column] = score

    # Save updated manifest
    save_manifest(df, manifest_path)

    return mask.sum()


def get_stage_progress(manifest_path: Path, stage: str) -> dict:
    """
    Get progress statistics for a stage.

    Args:
        manifest_path: Path to manifest.parquet
        stage: Stage name

    Returns:
        Dict with 'total', 'completed', 'pending', 'percent' keys
    """
    config = STAGE_CONFIG[stage]
    df = load_manifest(manifest_path)

    total = len(df)

    if config['depends_on']:
        # Only count items where dependency is met
        eligible = df[df[config['depends_on']] == True]
        total = len(eligible)
    else:
        eligible = df

    if config['status_column']:
        completed = eligible[config['status_column']].sum()
    else:
        completed = 0

    pending = total - completed
    percent = (completed / total * 100) if total > 0 else 0

    return {
        'total': total,
        'completed': int(completed),
        'pending': pending,
        'percent': percent,
    }
