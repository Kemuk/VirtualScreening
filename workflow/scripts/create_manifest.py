#!/usr/bin/env python3
"""
create_manifest.py

Generate the master manifest (Parquet) from targets.yaml configuration.
Scans the filesystem for existing SMILES files and outputs, creating a
comprehensive tracking table with 29 columns.

This manifest serves as the single source of truth for pipeline state.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import yaml
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from tqdm import tqdm


# 29-column manifest schema
MANIFEST_SCHEMA = pa.schema([
    # Identity (5 columns)
    ('ligand_id', pa.string()),
    ('protein_id', pa.string()),
    ('dataset', pa.string()),
    ('compound_key', pa.string()),
    ('is_active', pa.bool_()),

    # Chemistry (2 columns)
    ('smiles_input', pa.string()),
    ('smiles_canonical', pa.string()),

    # Input Sources (3 columns)
    ('source_smiles_file', pa.string()),
    ('source_sdf_path', pa.string()),
    ('source_mol2_path', pa.string()),

    # Preparation (2 columns)
    ('preparation_status', pa.bool_()),
    ('ligand_pdbqt_path', pa.string()),

    # Receptor (2 columns)
    ('receptor_pdbqt_path', pa.string()),
    ('receptor_pdb_path', pa.string()),

    # Docking Box (6 columns)
    ('box_center_x', pa.float64()),
    ('box_center_y', pa.float64()),
    ('box_center_z', pa.float64()),
    ('box_size_x', pa.float64()),
    ('box_size_y', pa.float64()),
    ('box_size_z', pa.float64()),

    # Docking (4 columns)
    ('docking_status', pa.bool_()),
    ('docked_pdbqt_path', pa.string()),
    ('docking_log_path', pa.string()),
    ('vina_score', pa.float64()),

    # Rescoring (3 columns)
    ('rescoring_status', pa.bool_()),
    ('docked_sdf_path', pa.string()),
    ('aev_plig_score', pa.float64()),

    # Metadata (2 columns)
    ('created_at', pa.timestamp('ns')),
    ('last_updated', pa.timestamp('ns')),
])


def load_config(config_path: Path) -> Dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Convert SMILES to canonical form using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def parse_smiles_file(smiles_path: Path) -> List[Tuple[str, str]]:
    """
    Parse SMILES file and return list of (ligand_id, smiles).
    Expects format: SMILES [whitespace] ligand_id
    """
    ligands = []
    if not smiles_path.exists():
        return ligands

    with open(smiles_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 2:
                print(f"Warning: Skipping malformed line {line_num} in {smiles_path}", file=sys.stderr)
                continue

            smiles = parts[0]
            ligand_id = parts[1]
            ligands.append((ligand_id, smiles))

    return ligands


def generate_manifest_entries(
    targets_config: Dict,
    workflow_config: Dict,
    project_root: Path,
) -> List[Dict]:
    """
    Generate manifest entries for all targets using parallel processing.

    Returns a list of dictionaries, one per ligand.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import cpu_count

    dataset_name = workflow_config.get('dataset', 'LIT_PCBA')
    default_box_size = workflow_config.get('default_box_size', {'x': 25.0, 'y': 25.0, 'z': 25.0})
    targets = targets_config.get('targets', {})

    # Get mode and limits
    mode = workflow_config.get('mode', 'test')
    if mode == 'test':
        max_actives = workflow_config.get('test', {}).get('actives_per_protein', 100)
        max_inactives = workflow_config.get('test', {}).get('inactives_per_protein', 9900)
    else:
        max_actives = None  # No limit
        max_inactives = None

    # Prepare all ligand tasks (combining actives and inactives)
    all_tasks = []
    for protein_id, target_config in targets.items():
        receptor_mol2 = project_root / target_config['receptor_mol2']
        actives_smi = project_root / target_config['actives_smi']
        inactives_smi = project_root / target_config['inactives_smi']

        box_center = target_config['box_center']
        box_size = target_config.get('box_size', default_box_size)

        target_dir = receptor_mol2.parent
        # Use {protein_id}_protein naming for receptor outputs
        receptor_pdbqt = target_dir / f"{protein_id}_protein.pdbqt"
        receptor_pdb = target_dir / f"{protein_id}_protein.pdb"

        # Combine actives and inactives into one list, with filtering for test mode
        ligands = []

        # Load actives (with limit if in test mode)
        actives_count = 0
        for ligand_id, smiles in parse_smiles_file(actives_smi):
            if max_actives is None or actives_count < max_actives:
                ligands.append((ligand_id, smiles, True))  # is_active=True
                actives_count += 1

        # Load inactives (with limit if in test mode)
        inactives_count = 0
        for ligand_id, smiles in parse_smiles_file(inactives_smi):
            if max_inactives is None or inactives_count < max_inactives:
                ligands.append((ligand_id, smiles, False))  # is_active=False
                inactives_count += 1

        # Create tasks for this target
        for ligand_id, smiles, is_active in ligands:
            source_file = actives_smi if is_active else inactives_smi
            task = {
                'ligand_id': ligand_id,
                'protein_id': protein_id,
                'dataset': dataset_name,
                'smiles': smiles,
                'is_active': is_active,
                'source_smiles_file': str(source_file.relative_to(project_root)),
                'source_mol2_path': str(receptor_mol2.relative_to(project_root)),
                'target_dir': target_dir,
                'receptor_pdbqt': receptor_pdbqt,
                'receptor_pdb': receptor_pdb,
                'box_center': box_center,
                'box_size': box_size,
                'project_root': project_root,
            }
            all_tasks.append(task)

    print(f"Processing {len(all_tasks)} ligands across {len(targets)} targets...")

    # Process in parallel
    max_workers = min(cpu_count(), 16)  # Cap at 16 workers
    entries = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(create_ligand_entry_wrapper, task): task
            for task in all_tasks
        }

        # Process results with progress bar
        with tqdm(total=len(all_tasks), desc="Creating entries", unit="ligand") as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    entry = future.result()
                    entries.append(entry)
                    # Update progress bar with current ligand info
                    pbar.set_postfix({
                        'target': task['protein_id'],
                        'ligand': task['ligand_id'][:15]  # Truncate long IDs
                    })
                    pbar.update(1)
                except Exception as e:
                    print(f"\nERROR processing {task['protein_id']}_{task['ligand_id']}: {e}", file=sys.stderr)
                    pbar.update(1)

    return entries


def create_ligand_entry_wrapper(task: Dict) -> Dict:
    """
    Wrapper function for parallel processing.

    Unpacks task dict and calls create_ligand_entry.
    """
    return create_ligand_entry(
        ligand_id=task['ligand_id'],
        protein_id=task['protein_id'],
        dataset=task['dataset'],
        smiles=task['smiles'],
        is_active=task['is_active'],
        source_smiles_file=task['source_smiles_file'],
        source_mol2_path=task['source_mol2_path'],
        target_dir=task['target_dir'],
        receptor_pdbqt=task['receptor_pdbqt'],
        receptor_pdb=task['receptor_pdb'],
        box_center=task['box_center'],
        box_size=task['box_size'],
        project_root=task['project_root'],
    )


def create_ligand_entry(
    ligand_id: str,
    protein_id: str,
    dataset: str,
    smiles: str,
    is_active: bool,
    source_smiles_file: str,
    source_mol2_path: str,
    target_dir: Path,
    receptor_pdbqt: Path,
    receptor_pdb: Path,
    box_center: Dict,
    box_size: Dict,
    project_root: Path,
) -> Dict:
    """
    Create a single manifest entry for a ligand.

    Checks filesystem to set status flags based on existing outputs.
    """
    compound_key = f"{protein_id}_{ligand_id}"
    smiles_canonical = canonicalize_smiles(smiles)

    # Determine subdirectory (actives or inactives)
    subdir = "actives" if is_active else "inactives"

    # Define expected paths
    ligand_pdbqt_path = target_dir / f"pdbqt/{subdir}/{ligand_id}.pdbqt"
    docked_pdbqt_path = target_dir / f"docked_vina/{subdir}/{ligand_id}_docked.pdbqt"
    docking_log_path = target_dir / f"docked_vina/{subdir}/log/{ligand_id}.log"
    docked_sdf_path = target_dir / f"docked_sdf/{subdir}/{ligand_id}.sdf"

    # Check preparation status
    preparation_status = ligand_pdbqt_path.exists()

    # Check docking status
    docking_status = docked_pdbqt_path.exists()
    vina_score = extract_vina_score(docking_log_path) if docking_status else None

    # Check rescoring status
    rescoring_status = False  # Will be updated by rescoring stage
    aev_plig_score = None

    # Convert paths to relative (from project root)
    now = datetime.now()

    return {
        # Identity
        'ligand_id': ligand_id,
        'protein_id': protein_id,
        'dataset': dataset,
        'compound_key': compound_key,
        'is_active': is_active,

        # Chemistry
        'smiles_input': smiles,
        'smiles_canonical': smiles_canonical,

        # Input Sources
        'source_smiles_file': source_smiles_file,
        'source_sdf_path': None,  # Not used for LIT_PCBA
        'source_mol2_path': source_mol2_path,  # Receptor MOL2 from targets.yaml

        # Preparation
        'preparation_status': preparation_status,
        'ligand_pdbqt_path': str(ligand_pdbqt_path.relative_to(project_root)) if preparation_status else str(ligand_pdbqt_path.relative_to(project_root)),

        # Receptor
        'receptor_pdbqt_path': str(receptor_pdbqt.relative_to(project_root)),
        'receptor_pdb_path': str(receptor_pdb.relative_to(project_root)),

        # Docking Box
        'box_center_x': float(box_center['x']),
        'box_center_y': float(box_center['y']),
        'box_center_z': float(box_center['z']),
        'box_size_x': float(box_size['x']),
        'box_size_y': float(box_size['y']),
        'box_size_z': float(box_size['z']),

        # Docking
        'docking_status': docking_status,
        'docked_pdbqt_path': str(docked_pdbqt_path.relative_to(project_root)),
        'docking_log_path': str(docking_log_path.relative_to(project_root)),
        'vina_score': vina_score,

        # Rescoring
        'rescoring_status': rescoring_status,
        'docked_sdf_path': str(docked_sdf_path.relative_to(project_root)),
        'aev_plig_score': aev_plig_score,

        # Metadata
        'created_at': now,
        'last_updated': now,
    }


def extract_vina_score(log_path: Path) -> Optional[float]:
    """
    Extract binding affinity from Vina log file.

    Looks for lines like:
       1        -8.5      0.000      0.000

    Returns the first (best) score.
    """
    if not log_path.exists():
        return None

    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('1 ') or line.startswith('1\t'):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            return float(parts[1])
                        except ValueError:
                            continue
    except Exception:
        pass

    return None


def save_manifest(entries: List[Dict], output_path: Path, backup: bool = True):
    """
    Save manifest to Parquet file with optional backup.

    Creates timestamped backup if manifest already exists.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create backup if file exists
    if backup and output_path.exists():
        backup_dir = output_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"manifest_{timestamp}.parquet"
        output_path.rename(backup_path)
        print(f"Created backup: {backup_path}")

    # Convert to DataFrame
    df = pd.DataFrame(entries)

    # Ensure correct data types
    df['is_active'] = df['is_active'].astype(bool)
    df['preparation_status'] = df['preparation_status'].astype(bool)
    df['docking_status'] = df['docking_status'].astype(bool)
    df['rescoring_status'] = df['rescoring_status'].astype(bool)

    # Convert timestamps
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['last_updated'] = pd.to_datetime(df['last_updated'])

    # Write to Parquet with compression
    table = pa.Table.from_pandas(df, schema=MANIFEST_SCHEMA)
    pq.write_table(table, output_path, compression='snappy')

    print(f"Saved manifest: {output_path}")
    print(f"  Total entries: {len(entries)}")
    print(f"  Prepared: {df['preparation_status'].sum()}")
    print(f"  Docked: {df['docking_status'].sum()}")
    print(f"  Rescored: {df['rescoring_status'].sum()}")


def filter_test_mode(entries: List[Dict], workflow_config: Dict) -> List[Dict]:
    """
    Filter entries for test mode.

    Takes a subset of ligands per protein for testing:
    - N actives per protein
    - N inactives per protein

    Args:
        entries: Full list of manifest entries
        workflow_config: Workflow configuration

    Returns:
        Filtered list of entries
    """
    test_config = workflow_config.get('test', {})
    actives_per_protein = test_config.get('actives_per_protein', 100)
    inactives_per_protein = test_config.get('inactives_per_protein', 9900)

    # Group by protein
    from collections import defaultdict
    by_protein = defaultdict(lambda: {'actives': [], 'inactives': []})

    for entry in entries:
        protein_id = entry['protein_id']
        if entry['is_active']:
            by_protein[protein_id]['actives'].append(entry)
        else:
            by_protein[protein_id]['inactives'].append(entry)

    # Select subset for each protein
    filtered_entries = []
    for protein_id, ligands in by_protein.items():
        # Take first N actives and inactives (deterministic)
        selected_actives = ligands['actives'][:actives_per_protein]
        selected_inactives = ligands['inactives'][:inactives_per_protein]

        filtered_entries.extend(selected_actives)
        filtered_entries.extend(selected_inactives)

        print(f"  {protein_id}: {len(selected_actives)} actives + {len(selected_inactives)} inactives = {len(selected_actives) + len(selected_inactives)} total")

    total_per_protein = actives_per_protein + inactives_per_protein
    print(f"\nTest mode: Filtered to {len(filtered_entries)} ligands (from {len(entries)} total)")
    print(f"  Targets: {len(by_protein)}")
    print(f"  Per target: up to {total_per_protein} ligands ({actives_per_protein} actives + {inactives_per_protein} inactives)")

    return filtered_entries


def main():
    parser = argparse.ArgumentParser(description="Generate master manifest from targets.yaml")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to main config file"
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("config/targets.yaml"),
        help="Path to targets config file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/master/manifest.parquet"),
        help="Output manifest path"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup of existing manifest"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing manifest (required if manifest exists)"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )

    args = parser.parse_args()

    # Check if manifest exists and overwrite flag is needed
    if args.output.exists() and not args.overwrite:
        print(f"ERROR: Manifest already exists at {args.output}", file=sys.stderr)
        print(f"Use --overwrite to replace it, or specify a different output path.", file=sys.stderr)
        sys.exit(1)

    # Load configurations
    workflow_config = load_config(args.config)
    targets_config = load_config(args.targets)

    # Generate manifest entries
    print("Generating manifest entries...")
    entries = generate_manifest_entries(
        targets_config=targets_config,
        workflow_config=workflow_config,
        project_root=args.project_root,
    )

    if not entries:
        print("ERROR: No entries generated. Check your configuration and SMILES files.", file=sys.stderr)
        sys.exit(1)

    # Apply test mode filtering if enabled
    mode = workflow_config.get('mode', 'test')
    if mode == 'test':
        entries = filter_test_mode(entries, workflow_config)

    # Save manifest
    save_manifest(entries, args.output, backup=not args.no_backup)
    print(f"✓ Manifest generation complete!")


if __name__ == "__main__":
    main()
