#!/usr/bin/env python3
"""
prepare_all_ligands.py

Batch preparation of all ligands from SMILES to PDBQT format.
Uses parallel processing with batching for efficiency.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import pandas as pd
from tqdm import tqdm

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import conversion function from smi2pdbqt
from smi2pdbqt import smiles_to_pdbqt


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load manifest from Parquet file."""
    return pd.read_parquet(manifest_path)


def prepare_single_ligand(task: Dict) -> Dict:
    """
    Prepare a single ligand.

    Args:
        task: Dictionary with ligand information

    Returns:
        Dictionary with result (success status, ligand info)
    """
    ligand_id = task['ligand_id']
    protein_id = task['protein_id']
    smiles = task['smiles']
    pdbqt_path = Path(task['pdbqt_path'])
    ph = task['ph']
    partial_charge = task['partial_charge']

    try:
        success = smiles_to_pdbqt(
            smiles=smiles,
            pdbqt_path=pdbqt_path,
            ph=ph,
            partial_charge=partial_charge,
        )

        return {
            'ligand_id': ligand_id,
            'protein_id': protein_id,
            'success': success,
            'error': None
        }
    except Exception as e:
        return {
            'ligand_id': ligand_id,
            'protein_id': protein_id,
            'success': False,
            'error': str(e)
        }


def prepare_ligand_batch(batch: List[Dict]) -> Dict:
    """
    Prepare a batch of ligands (worker function for parallel processing).

    Args:
        batch: List of ligand task dictionaries

    Returns:
        Dictionary with batch results
    """
    results = []
    for task in batch:
        result = prepare_single_ligand(task)
        results.append(result)

    succeeded = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])
    failed_ligands = [r for r in results if not r['success']]

    return {
        'batch_size': len(batch),
        'succeeded': succeeded,
        'failed': failed,
        'failed_ligands': failed_ligands,
        'last_protein': batch[-1]['protein_id'],
        'last_ligand': batch[-1]['ligand_id']
    }


def prepare_all_ligands(
    manifest_path: Path,
    project_root: Path,
    ph: float = 7.4,
    partial_charge: str = "gasteiger",
    max_workers: int = None,
    batch_size: int = 100,
    force: bool = False,
) -> Dict[str, int]:
    """
    Prepare all ligands from manifest using parallel batch processing.

    Args:
        manifest_path: Path to manifest Parquet file
        project_root: Project root directory
        ph: pH for protonation
        partial_charge: Charge calculation method
        max_workers: Maximum parallel workers (default: CPU count, capped at 16)
        batch_size: Number of ligands per batch (default: 100)
        force: Force re-preparation of existing files

    Returns:
        Dictionary with statistics (total, succeeded, failed, skipped)
    """
    # Load manifest
    print(f"Loading manifest: {manifest_path}")
    manifest = load_manifest(manifest_path)

    print(f"Total ligands in manifest: {len(manifest)}")
    print(f"  Actives: {manifest['is_active'].sum()}")
    print(f"  Inactives: {(~manifest['is_active']).sum()}")

    # Build task list
    tasks = []
    for _, row in manifest.iterrows():
        pdbqt_path = project_root / row['ligand_pdbqt_path']

        # Skip if already prepared (unless force=True)
        if not force and pdbqt_path.exists():
            continue

        task = {
            'ligand_id': row['ligand_id'],
            'protein_id': row['protein_id'],
            'smiles': row['smiles_input'],
            'pdbqt_path': str(pdbqt_path),
            'ph': ph,
            'partial_charge': partial_charge,
        }
        tasks.append(task)

    if not tasks:
        print("✓ All ligands already prepared!")
        return {
            'total': len(manifest),
            'succeeded': 0,
            'failed': 0,
            'skipped': len(manifest)
        }

    print(f"\nPreparing {len(tasks)} ligands...")
    if not force:
        print(f"  ({len(manifest) - len(tasks)} already exist, skipping)")

    # Split tasks into batches
    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
    print(f"  Processing in {len(batches)} batches of ~{batch_size} ligands")

    # Determine worker count
    if max_workers is None:
        max_workers = min(cpu_count(), 16)
    print(f"  Using {max_workers} parallel workers\n")

    # Process batches in parallel with progress bar
    succeeded = 0
    failed = 0
    all_failed_ligands = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(prepare_ligand_batch, batch): batch
            for batch in batches
        }

        # Process results with progress bar
        with tqdm(total=len(tasks), desc="Preparing ligands", unit=" ligand", ncols=100) as pbar:
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    result = future.result()

                    succeeded += result['succeeded']
                    failed += result['failed']
                    all_failed_ligands.extend(result['failed_ligands'])

                    # Update progress bar
                    pbar.set_postfix({
                        'target': result['last_protein'][:8],
                        'ligand': result['last_ligand'][:12],
                        'ok': succeeded,
                        'fail': failed
                    })
                    pbar.update(result['batch_size'])

                except Exception as e:
                    # If entire batch fails, count all as failed
                    failed += len(batch)
                    for task in batch:
                        all_failed_ligands.append({
                            'ligand_id': task['ligand_id'],
                            'protein_id': task['protein_id'],
                            'error': f'Batch error: {str(e)}'
                        })
                    pbar.update(len(batch))

    # Print summary
    print(f"\n{'='*60}")
    print(f"Preparation complete!")
    print(f"  Total processed: {len(tasks)}")
    print(f"  ✓ Succeeded: {succeeded}")
    print(f"  ✗ Failed: {failed}")
    print(f"  Skipped: {len(manifest) - len(tasks)}")
    print(f"{'='*60}")

    # Print failed ligands if any
    if all_failed_ligands:
        print(f"\nFailed ligands ({len(all_failed_ligands)}):")
        for item in all_failed_ligands[:10]:  # Show first 10
            print(f"  {item['protein_id']}_{item['ligand_id']}: {item['error']}")
        if len(all_failed_ligands) > 10:
            print(f"  ... and {len(all_failed_ligands) - 10} more")

    return {
        'total': len(manifest),
        'succeeded': succeeded,
        'failed': failed,
        'skipped': len(manifest) - len(tasks)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch preparation of ligands (SMILES → PDBQT)"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest Parquet file"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--ph",
        type=float,
        default=7.4,
        help="pH for protonation (default: 7.4)"
    )
    parser.add_argument(
        "--partial-charge",
        type=str,
        default="gasteiger",
        choices=["gasteiger", "mmff94", "eem"],
        help="Partial charge method (default: gasteiger)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum parallel workers (default: CPU count, capped at 16)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of ligands per batch (default: 100)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-preparation of existing files"
    )

    args = parser.parse_args()

    # Check manifest exists
    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    # Run preparation
    stats = prepare_all_ligands(
        manifest_path=args.manifest,
        project_root=args.project_root,
        ph=args.ph,
        partial_charge=args.partial_charge,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        force=args.force,
    )

    # Exit with error if any failures
    if stats['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
