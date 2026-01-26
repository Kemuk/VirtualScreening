#!/usr/bin/env python3
"""
create_manifest.py

Generate the master manifest (Parquet) from targets.yaml configuration.
Scans the filesystem for existing SMILES files and outputs, creating a
comprehensive tracking table with 40 columns.

This manifest serves as the single source of truth for pipeline state.
"""

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import yaml
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from tqdm import tqdm


# 40-column manifest schema
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

    # Conversion (1 column)
    ('conversion_status', pa.bool_()),

    # Rescoring (14 columns)
    ('rescoring_status', pa.bool_()),
    ('docked_sdf_path', pa.string()),
    ('binding_affinity_pK', pa.float64()),
    ('aev_plig_best_score', pa.float64()),
    ('aev_prediction_0', pa.float64()),
    ('aev_prediction_1', pa.float64()),
    ('aev_prediction_2', pa.float64()),
    ('aev_prediction_3', pa.float64()),
    ('aev_prediction_4', pa.float64()),
    ('aev_prediction_5', pa.float64()),
    ('aev_prediction_6', pa.float64()),
    ('aev_prediction_7', pa.float64()),
    ('aev_prediction_8', pa.float64()),
    ('aev_prediction_9', pa.float64()),

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

    mode = workflow_config.get('mode', 'test')
    mode_config = workflow_config.get(mode, {})
    max_actives = mode_config.get('actives_per_protein')
    max_inactives = mode_config.get('inactives_per_protein')
    max_items = None
    if mode == "devel":
        max_items = (
            workflow_config.get("chunking", {})
            .get("devel", {})
            .get("max_items")
        )

    # Prepare all ligand tasks (combining actives and inactives)
    all_tasks = []
    stop_early = False
    for protein_id, target_config in targets.items():
        if max_items is not None and len(all_tasks) >= max_items:
            stop_early = True
            break
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
            if max_items is not None and len(all_tasks) >= max_items:
                stop_early = True
                break
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
        if stop_early:
            break

    if max_items is not None:
        print(
            f"Processing {len(all_tasks)} ligands across {len(targets)} targets "
            f"(max_items={max_items}, mode={mode})..."
        )
    else:
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
    vina_score = extract_vina_score(docked_pdbqt_path) if docking_status else None

    # Check conversion status (SDF exists)
    conversion_status = docked_sdf_path.exists()

    # Check rescoring status
    rescoring_status = False  # Will be updated by rescoring stage
    binding_affinity_pK = None
    aev_plig_best_score = None
    aev_predictions = [None] * 10  # aev_prediction_0 through aev_prediction_9

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

        # Conversion
        'conversion_status': conversion_status,

        # Rescoring
        'rescoring_status': rescoring_status,
        'docked_sdf_path': str(docked_sdf_path.relative_to(project_root)),
        'binding_affinity_pK': binding_affinity_pK,
        'aev_plig_best_score': aev_plig_best_score,
        'aev_prediction_0': aev_predictions[0],
        'aev_prediction_1': aev_predictions[1],
        'aev_prediction_2': aev_predictions[2],
        'aev_prediction_3': aev_predictions[3],
        'aev_prediction_4': aev_predictions[4],
        'aev_prediction_5': aev_predictions[5],
        'aev_prediction_6': aev_predictions[6],
        'aev_prediction_7': aev_predictions[7],
        'aev_prediction_8': aev_predictions[8],
        'aev_prediction_9': aev_predictions[9],

        # Metadata
        'created_at': now,
        'last_updated': now,
    }


def process_batch(items: List[Dict], config: Dict) -> List[Dict]:
    """
    Process a batch of items for manifest creation (worker function).

    This is called by SLURM array workers to process their chunk.
    Each item is processed to create a manifest entry with:
    - Canonicalized SMILES (via RDKit)
    - Generated file paths
    - Status checks (file existence)

    Args:
        items: List of item dicts from scan_manifest_items.py
        config: Workflow configuration dict

    Returns:
        List of result dicts (one per successfully processed item)
    """
    from pathlib import Path
    from datetime import datetime

    results = []
    project_root = Path(config.get('project_root', '.')).resolve()

    for item in items:
        try:
            ligand_id = item['ligand_id']
            protein_id = item['protein_id']
            smiles = item['smiles']
            is_active = item['is_active']

            # Canonicalize SMILES
            smiles_canonical = canonicalize_smiles(smiles)

            # Build paths
            target_dir = project_root / item['target_dir']
            subdir = "actives" if is_active else "inactives"

            ligand_pdbqt_path = target_dir / f"pdbqt/{subdir}/{ligand_id}.pdbqt"
            docked_pdbqt_path = target_dir / f"docked_vina/{subdir}/{ligand_id}_docked.pdbqt"
            docking_log_path = target_dir / f"docked_vina/{subdir}/log/{ligand_id}.log"
            docked_sdf_path = target_dir / f"docked_sdf/{subdir}/{ligand_id}.sdf"
            receptor_pdbqt = target_dir / f"{protein_id}_protein.pdbqt"
            receptor_pdb = target_dir / f"{protein_id}_protein.pdb"

            # Check statuses
            preparation_status = ligand_pdbqt_path.exists()
            docking_status = docked_pdbqt_path.exists()
            vina_score = extract_vina_score(docked_pdbqt_path) if docking_status else None
            conversion_status = docked_sdf_path.exists()

            now = datetime.now().isoformat()

            result = {
                'ligand_id': ligand_id,
                'protein_id': protein_id,
                'dataset': item['dataset'],
                'compound_key': f"{protein_id}_{ligand_id}",
                'is_active': is_active,
                'smiles_input': smiles,
                'smiles_canonical': smiles_canonical,
                'source_smiles_file': item['source_smiles_file'],
                'source_sdf_path': None,
                'source_mol2_path': item['receptor_mol2'],
                'preparation_status': preparation_status,
                'ligand_pdbqt_path': str(ligand_pdbqt_path.relative_to(project_root)),
                'receptor_pdbqt_path': str(receptor_pdbqt.relative_to(project_root)),
                'receptor_pdb_path': str(receptor_pdb.relative_to(project_root)),
                'box_center_x': item['box_center_x'],
                'box_center_y': item['box_center_y'],
                'box_center_z': item['box_center_z'],
                'box_size_x': item['box_size_x'],
                'box_size_y': item['box_size_y'],
                'box_size_z': item['box_size_z'],
                'docking_status': docking_status,
                'docked_pdbqt_path': str(docked_pdbqt_path.relative_to(project_root)),
                'docking_log_path': str(docking_log_path.relative_to(project_root)),
                'vina_score': vina_score,
                'conversion_status': conversion_status,
                'rescoring_status': False,
                'docked_sdf_path': str(docked_sdf_path.relative_to(project_root)),
                'binding_affinity_pK': None,
                'aev_plig_best_score': None,
                'aev_prediction_0': None,
                'aev_prediction_1': None,
                'aev_prediction_2': None,
                'aev_prediction_3': None,
                'aev_prediction_4': None,
                'aev_prediction_5': None,
                'aev_prediction_6': None,
                'aev_prediction_7': None,
                'aev_prediction_8': None,
                'aev_prediction_9': None,
                'created_at': now,
                'last_updated': now,
                'success': True,
            }
            results.append(result)

        except Exception as e:
            # Record failure but continue processing
            results.append({
                'ligand_id': item.get('ligand_id', 'unknown'),
                'protein_id': item.get('protein_id', 'unknown'),
                'success': False,
                'error': str(e),
            })

    return results


def extract_vina_score(pdbqt_path: Path) -> Optional[float]:
    """
    Extract binding affinity from docked PDBQT file.

    Reads line 2 which contains the REMARK with Vina score:
        REMARK VINA RESULT:    -8.5      0.000      0.000

    Returns the first (best) score.
    """
    if not pdbqt_path.exists():
        return None

    try:
        with open(pdbqt_path) as f:
            next(f)  # Skip MODEL line
            line2 = next(f)
            match = re.search(r'-?\d+\.\d+', line2)
            if match:
                return float(match.group())
    except Exception:
        pass

    return None


def update_vina_scores(manifest_path: Path, project_root: Path, max_workers: int = 16):
    """
    Update missing vina_score values from docked PDBQT files.

    This is much faster than regenerating the entire manifest when you only
    need to backfill missing Vina scores.

    Args:
        manifest_path: Path to existing manifest parquet
        project_root: Project root directory for resolving paths
        max_workers: Number of parallel workers for extraction
    """
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    # Load existing manifest
    df = pl.read_parquet(manifest_path)
    total_rows = len(df)

    # Filter to docked but missing score
    needs_score = df.filter(
        (pl.col('docking_status') == True) &
        pl.col('vina_score').is_null()
    )

    if len(needs_score) == 0:
        print("No ligands need score extraction - all vina_score values are populated.")
        return

    print(f"Found {len(needs_score)} / {total_rows} ligands needing score extraction")

    # Build list of paths
    paths = [project_root / p for p in needs_score['docked_pdbqt_path'].to_list()]
    compound_keys = needs_score['compound_key'].to_list()

    # Parallel extraction
    print(f"Extracting scores using {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        scores = list(tqdm(
            executor.map(extract_vina_score, paths),
            total=len(paths),
            desc="Extracting scores"
        ))

    # Count successful extractions
    successful = sum(1 for s in scores if s is not None)
    print(f"Successfully extracted {successful} / {len(scores)} scores")

    # Build updates dataframe
    updates = pl.DataFrame({
        'compound_key': compound_keys,
        'vina_score_new': scores
    })

    # Join and coalesce
    df = df.join(updates, on='compound_key', how='left')
    df = df.with_columns(
        pl.coalesce(['vina_score_new', 'vina_score']).alias('vina_score')
    ).drop('vina_score_new')

    # Update last_updated timestamp for modified rows
    df = df.with_columns(
        pl.when(pl.col('compound_key').is_in(compound_keys))
        .then(pl.lit(datetime.now()))
        .otherwise(pl.col('last_updated'))
        .alias('last_updated')
    )

    # Save back
    df.write_parquet(manifest_path)
    print(f"Updated manifest: {manifest_path}")
    print(f"  Scores extracted: {successful}")
    print(f"  Scores missing: {len(scores) - successful}")


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

    df = pl.DataFrame(entries)
    if 'compound_key' in df.columns:
        before = df.height
        df = df.unique(subset=['compound_key'], keep='first')
        after = df.height
        removed = before - after
        if removed:
            print(f"Deduplicated {removed} entries by compound_key")

    df = df.with_columns([
        pl.col('is_active').cast(pl.Boolean),
        pl.col('preparation_status').cast(pl.Boolean),
        pl.col('docking_status').cast(pl.Boolean),
        pl.col('conversion_status').cast(pl.Boolean),
        pl.col('rescoring_status').cast(pl.Boolean),
        pl.col('created_at').cast(pl.Datetime),
        pl.col('last_updated').cast(pl.Datetime),
    ])

    # Write to Parquet with compression
    table = df.to_arrow()
    table = table.cast(MANIFEST_SCHEMA)
    pq.write_table(table, output_path, compression='snappy')

    print(f"Saved manifest: {output_path}")
    print(f"  Total entries: {len(entries)}")
    print(f"  Prepared: {df['preparation_status'].sum()}")
    print(f"  Docked: {df['docking_status'].sum()}")
    print(f"  Converted: {df['conversion_status'].sum()}")
    print(f"  Rescored: {df['rescoring_status'].sum()}")


def filter_mode_entries(entries: List[Dict], workflow_config: Dict) -> List[Dict]:
    """
    Filter entries for the configured mode if limits are set.
    """
    mode = workflow_config.get('mode', 'test')
    mode_config = workflow_config.get(mode, {})
    actives_per_protein = mode_config.get('actives_per_protein')
    inactives_per_protein = mode_config.get('inactives_per_protein')
    if actives_per_protein is None and inactives_per_protein is None:
        filtered_entries = entries
    else:
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
            actives_limit = actives_per_protein if actives_per_protein is not None else len(ligands['actives'])
            inactives_limit = inactives_per_protein if inactives_per_protein is not None else len(ligands['inactives'])
            selected_actives = ligands['actives'][:actives_limit]
            selected_inactives = ligands['inactives'][:inactives_limit]

            filtered_entries.extend(selected_actives)
            filtered_entries.extend(selected_inactives)

            print(
                f"  {protein_id}: {len(selected_actives)} actives + {len(selected_inactives)} inactives = "
                f"{len(selected_actives) + len(selected_inactives)} total"
            )

        total_per_protein = (actives_per_protein or 0) + (inactives_per_protein or 0)
        print(f"\nMode {mode}: Filtered to {len(filtered_entries)} ligands (from {len(entries)} total)")
        print(f"  Targets: {len(by_protein)}")
        if actives_per_protein is not None or inactives_per_protein is not None:
            print(
                f"  Per target: up to {total_per_protein} ligands "
                f"({actives_per_protein or 0} actives + {inactives_per_protein or 0} inactives)"
            )

    if mode == "devel":
        max_items = (
            workflow_config.get("chunking", {})
            .get("devel", {})
            .get("max_items")
        )
        if max_items:
            filtered_entries = filtered_entries[:max_items]
            print(f"  Devel mode: using first {len(filtered_entries)} ligands (max_items={max_items})")

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
        "--mode",
        help="Override workflow mode from config (e.g., devel, test, production)",
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
    parser.add_argument(
        "--update-vina-scores",
        action="store_true",
        help="Only update missing vina_score values from docked PDBQT files (much faster than full rebuild)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers for score extraction (default: 16)"
    )

    args = parser.parse_args()

    # Handle update-vina-scores mode
    if args.update_vina_scores:
        update_vina_scores(args.output, args.project_root, args.workers)
        print("✓ Vina score update complete!")
        return

    # Check if manifest exists and overwrite flag is needed
    if args.output.exists() and not args.overwrite:
        print(f"ERROR: Manifest already exists at {args.output}", file=sys.stderr)
        print(f"Use --overwrite to replace it, or specify a different output path.", file=sys.stderr)
        sys.exit(1)

    # Load configurations
    workflow_config = load_config(args.config)
    targets_config = load_config(args.targets)
    if args.mode:
        workflow_config["mode"] = args.mode

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

    entries = filter_mode_entries(entries, workflow_config)

    # Save manifest
    save_manifest(entries, args.output, backup=not args.no_backup)
    print(f"✓ Manifest generation complete!")


if __name__ == "__main__":
    main()
