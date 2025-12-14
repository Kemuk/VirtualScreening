"""
common.smk

Common functions and utilities for the virtual screening workflow.
Provides configuration loading and helper functions for manifest operations.
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path


# =============================================================================
# Configuration Loading
# =============================================================================

def load_workflow_config():
    """Load main workflow configuration."""
    return config


def get_targets():
    """Get list of all target IDs from targets.yaml."""
    targets_config = load_targets_config()
    return list(targets_config['targets'].keys())


def load_targets_config():
    """Load targets configuration from YAML."""
    import yaml
    targets_path = Path(config.get('targets_config', 'config/targets.yaml'))
    with open(targets_path) as f:
        return yaml.safe_load(f)


def get_target_config(target_id):
    """Get configuration for a specific target."""
    targets = load_targets_config()
    return targets['targets'][target_id]


# =============================================================================
# Manifest Operations
# =============================================================================

def load_manifest(manifest_path=None):
    """
    Load manifest as pandas DataFrame.

    Args:
        manifest_path: Path to manifest (default: from config)

    Returns:
        pandas.DataFrame
    """
    if manifest_path is None:
        manifest_path = Path(config['manifest_dir']) / 'manifest.parquet'

    table = pq.read_table(manifest_path)
    return table.to_pandas()


def get_manifest_entries(manifest_df, **filters):
    """
    Filter manifest entries by criteria.

    Args:
        manifest_df: Manifest DataFrame
        **filters: Column filters (e.g., protein_id='ADRB2', preparation_status=False)

    Returns:
        Filtered DataFrame
    """
    df = manifest_df.copy()

    for column, value in filters.items():
        if column not in df.columns:
            raise ValueError(f"Unknown column: {column}")
        df = df[df[column] == value]

    return df


def get_compound_keys(manifest_df, **filters):
    """
    Get list of compound keys matching filters.

    Args:
        manifest_df: Manifest DataFrame
        **filters: Column filters

    Returns:
        List of compound keys
    """
    filtered = get_manifest_entries(manifest_df, **filters)
    return filtered['compound_key'].tolist()


def get_ligand_paths(manifest_df, path_column, **filters):
    """
    Get list of file paths for ligands matching filters.

    Args:
        manifest_df: Manifest DataFrame
        path_column: Column name containing paths (e.g., 'ligand_pdbqt_path')
        **filters: Column filters

    Returns:
        List of paths
    """
    filtered = get_manifest_entries(manifest_df, **filters)
    return filtered[path_column].tolist()


# =============================================================================
# Path Helpers
# =============================================================================

def get_box_params(target_id):
    """
    Get docking box parameters for a target.

    Returns:
        dict with keys: center_x, center_y, center_z, size_x, size_y, size_z
    """
    target_config = get_target_config(target_id)
    default_size = config.get('default_box_size', {'x': 25.0, 'y': 25.0, 'z': 25.0})

    box_center = target_config['box_center']
    box_size = target_config.get('box_size', default_size)

    return {
        'center_x': box_center['x'],
        'center_y': box_center['y'],
        'center_z': box_center['z'],
        'size_x': box_size['x'],
        'size_y': box_size['y'],
        'size_z': box_size['z'],
    }


def get_receptor_paths(target_id):
    """
    Get receptor file paths for a target.

    Returns:
        dict with keys: mol2, pdbqt, pdb
    """
    target_config = get_target_config(target_id)
    receptor_mol2 = Path(target_config['receptor_mol2'])
    target_dir = receptor_mol2.parent

    return {
        'mol2': str(receptor_mol2),
        'pdbqt': str(target_dir / f'{target_id}_protein.pdbqt'),
        'pdb': str(target_dir / f'{target_id}_protein.pdb'),
    }


def get_smiles_files(target_id):
    """
    Get SMILES file paths for a target.

    Returns:
        dict with keys: actives, inactives
    """
    target_config = get_target_config(target_id)
    return {
        'actives': target_config['actives_smi'],
        'inactives': target_config['inactives_smi'],
    }


# =============================================================================
# Resource Helpers
# =============================================================================

def get_resources(rule_name):
    """
    Get resource requirements for a rule from config.

    Args:
        rule_name: Name of the rule (e.g., 'preparation', 'docking_gpu')

    Returns:
        dict with resource specifications
    """
    resources = config.get('resources', {})
    rule_resources = resources.get(rule_name, {}).copy()

    # Add SLURM-specific flags for cluster profile
    partition = rule_resources.get('partition', 'htc')
    rule_resources['partition_flag'] = f"--partition={partition}" if partition else ""

    # Add GPU flag if needed
    gpus = rule_resources.get('gpus', 0)
    rule_resources['gpu_flag'] = f"--gres=gpu:{gpus}" if gpus > 0 else ""

    # Format runtime as HH:MM:SS for SLURM
    time_min = rule_resources.get('time_min', 60)
    hours = time_min // 60
    mins = time_min % 60
    rule_resources['runtime'] = f"{hours:02d}:{mins:02d}:00"

    return rule_resources


def get_tool_path(tool_name):
    """
    Get path to external tool from config.

    Args:
        tool_name: Tool identifier (e.g., 'vina_gpu', 'obabel')

    Returns:
        str: Path to tool executable
    """
    tools = config.get('tools', {})
    return tools.get(tool_name, tool_name)  # Default to tool name if not in config


# =============================================================================
# Validation Helpers
# =============================================================================

def validate_manifest_exists():
    """Check if manifest file exists, raise error if not."""
    manifest_path = Path(config['manifest_dir']) / 'manifest.parquet'
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run: python workflow/scripts/create_manifest.py"
        )


def validate_target_exists(target_id):
    """Check if target exists in configuration."""
    targets = get_targets()
    if target_id not in targets:
        raise ValueError(f"Unknown target: {target_id}. Available: {targets}")


# =============================================================================
# Checkpoint Functions
# =============================================================================

def checkpoint_manifest():
    """
    Checkpoint function to reload manifest and determine which files to process.

    This is used by Snakemake to dynamically determine DAG based on manifest state.
    """
    validate_manifest_exists()
    return load_manifest()
