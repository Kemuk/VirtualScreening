#!/usr/bin/env python3
"""
prepare_all_ligands.py

Batch preparation of all ligands from SMILES to PDBQT format.
Uses parallel processing with a single progress bar (like manifest generation).
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


def prepare_ligand_task(task: Dict) -> Dict:
    """
    Prepare a single ligand (worker function for parallel processing).

    Args:
        task: Dictionary with ligand information

    Returns:
        Dictionary with results (success status, ligand info)
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
            'pdbqt_path': str(pdbqt_path),
            'success': success,
            'error': None
        }
    except Exception as e:
        return {
            'ligand_id': ligand_id,
            'protein_id': protein_id,
            'pdbqt_path': str(pdbqt_path),
            'success': False,
            'error': str(e)
        }


def prepare_all_ligands(
    manifest_path: Path,
    project_root: Path,
    ph: float = 7.4,
    partial_charge: str = "gasteiger",
    max_workers: int = None,
    force: bool = False,
) -> Dict[str, int]:
    """
    Prepare all ligands from manifest using parallel processing.

    Args:
        manifest_path: Path to manifest Parquet file
        project_root: Project root directory
        ph: pH for protonation
        partial_charge: Charge calculation method
        max_workers: Maximum parallel workers (default: CPU count, capped at 16)
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

    # Determine worker count
    if max_workers is None:
        max_workers = min(cpu_count(), 16)
    print(f"  Using {max_workers} parallel workers\n")

    # Process in parallel with progress bar
    succeeded = 0
    failed = 0
    failed_ligands = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(prepare_ligand_task, task): task
            for task in tasks
        }

        # Process results with progress bar
        with tqdm(total=len(tasks), desc="Preparing ligands", unit=" ligand", ncols=100) as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()

                    if result['success']:
                        succeeded += 1
                    else:
                        failed += 1
                        failed_ligands.append({
                            'ligand_id': result['ligand_id'],
                            'protein_id': result['protein_id'],
                            'error': result['error']
                        })

                    # Update progress bar with current ligand info
                    pbar.set_postfix({
                        'target': task['protein_id'][:8],
                        'ligand': task['ligand_id'][:12],
                        'ok': succeeded,
                        'fail': failed
                    })
                    pbar.update(1)

                except Exception as e:
                    failed += 1
                    failed_ligands.append({
                        'ligand_id': task['ligand_id'],
                        'protein_id': task['protein_id'],
                        'error': str(e)
                    })
                    pbar.update(1)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Preparation complete!")
    print(f"  Total processed: {len(tasks)}")
    print(f"  ✓ Succeeded: {succeeded}")
    print(f"  ✗ Failed: {failed}")
    print(f"  Skipped: {len(manifest) - len(tasks)}")
    print(f"{'='*60}")

    # Print failed ligands if any
    if failed_ligands:
        print(f"\nFailed ligands ({len(failed_ligands)}):")
        for item in failed_ligands[:10]:  # Show first 10
            print(f"  {item['protein_id']}_{item['ligand_id']}: {item['error']}")
        if len(failed_ligands) > 10:
            print(f"  ... and {len(failed_ligands) - 10} more")

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
        force=args.force,
    )

    # Exit with error if any failures
    if stats['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
