#!/usr/bin/env python3
"""
ligands.py - Worker for SMILES to PDBQT conversion.

Usage:
    python -m workflow.slurm.workers.ligands \
        --pending data/master/pending/ligands.parquet \
        --task-id 0 \
        --num-chunks 500
"""

import argparse
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

from workflow.slurm.workers import read_slice, write_results

# Import conversion function
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from smi2pdbqt import smiles_to_pdbqt


def process_item(row: dict, ph: float, partial_charge: str) -> dict:
    """
    Process a single ligand: SMILES to PDBQT.

    Args:
        row: Row from pending parquet (as dict)
        ph: pH for protonation
        partial_charge: Charge calculation method

    Returns:
        Result dict with compound_key, ligand_id, paths, success, error
    """
    compound_key = row['compound_key']
    ligand_id = row.get('ligand_id', '')
    smiles = row.get('smiles_input', '')
    ligand_pdbqt_path = row.get('ligand_pdbqt_path', '')

    pdbqt_path = Path(ligand_pdbqt_path) if ligand_pdbqt_path else None

    # Base result with identifying info and paths
    base_result = {
        'compound_key': compound_key,
        'ligand_id': ligand_id,
        'smiles': smiles,
        'ligand_pdbqt_path': ligand_pdbqt_path,
    }

    # Skip if already exists
    if pdbqt_path and pdbqt_path.exists():
        return {
            **base_result,
            'success': True,
            'skipped': True,
            'error': '',
        }

    if not pdbqt_path:
        return {
            **base_result,
            'success': False,
            'skipped': False,
            'error': 'No ligand_pdbqt_path specified',
        }

    try:
        # Create output directory
        pdbqt_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert SMILES to PDBQT
        success = smiles_to_pdbqt(
            smiles=smiles,
            pdbqt_path=pdbqt_path,
            ph=ph,
            partial_charge=partial_charge,
        )

        return {
            **base_result,
            'success': success,
            'skipped': False,
            'error': '' if success else 'SMILES to PDBQT conversion failed',
        }

    except Exception as e:
        return {
            **base_result,
            'success': False,
            'skipped': False,
            'error': str(e),
        }


def process_slice(
    pending_path: Path,
    task_id: int,
    num_chunks: int,
    config_path: Path,
    results_dir: Path,
) -> int:
    """
    Process a slice of ligands for this worker.

    Args:
        pending_path: Path to pending parquet
        task_id: SLURM_ARRAY_TASK_ID
        num_chunks: Total number of chunks
        config_path: Path to config.yaml
        results_dir: Directory for result CSV files

    Returns:
        Number of items processed
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    prep_config = config.get('preparation', {})
    ph = prep_config.get('ph', 7.4)
    partial_charge = prep_config.get('partial_charge', 'gasteiger')

    # Read slice
    df = read_slice(pending_path, task_id, num_chunks)

    if df.empty:
        print(f"Task {task_id}: Empty slice, nothing to do")
        return 0

    print(f"Task {task_id}: Processing {len(df)} ligands")

    # Process each item with progress bar
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Task {task_id}"):
        result = process_item(row.to_dict(), ph, partial_charge)
        results.append(result)

    # Write results
    output_path = write_results(results, results_dir, 'ligands', task_id)
    print(f"Task {task_id}: Wrote {len(results)} results to {output_path}")

    # Summary
    succeeded = sum(1 for r in results if r.get('success'))
    failed = sum(1 for r in results if not r.get('success'))
    skipped = sum(1 for r in results if r.get('skipped'))

    print(f"Task {task_id}: {succeeded} succeeded, {failed} failed, {skipped} skipped")

    return len(results)


def main():
    parser = argparse.ArgumentParser(
        description="Worker for SMILES to PDBQT conversion"
    )
    parser.add_argument(
        "--pending",
        type=Path,
        required=True,
        help="Path to pending parquet file"
    )
    parser.add_argument(
        "--task-id",
        type=int,
        required=True,
        help="SLURM_ARRAY_TASK_ID"
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        required=True,
        help="Total number of chunks"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/master/results"),
        help="Directory for result CSV files"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.pending.exists():
        print(f"ERROR: Pending file not found: {args.pending}", file=sys.stderr)
        sys.exit(1)

    if not args.config.exists():
        print(f"ERROR: Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Process slice
    num_processed = process_slice(
        pending_path=args.pending,
        task_id=args.task_id,
        num_chunks=args.num_chunks,
        config_path=args.config,
        results_dir=args.results_dir,
    )

    print(f"Task {args.task_id}: Done. Processed {num_processed} items.")
    sys.exit(0)


if __name__ == "__main__":
    main()
