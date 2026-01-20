#!/usr/bin/env python3
"""
conversion.py - Worker for PDBQT to SDF conversion.

Usage:
    python -m workflow.slurm.workers.conversion \
        --pending data/master/pending/conversion.parquet \
        --task-id 0 \
        --num-chunks 500
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from workflow.slurm.workers import read_slice, write_results


def extract_model_from_pdbqt(pdbqt_path: Path, model_index: int = 0) -> str:
    """
    Extract a specific model from multi-model PDBQT file.

    Args:
        pdbqt_path: Path to docked PDBQT file
        model_index: Model index to extract (0 = first/best)

    Returns:
        PDBQT content for the selected model as string
    """
    models = []
    current_model = []
    in_model = False

    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith('MODEL'):
                in_model = True
                current_model = [line]
            elif line.startswith('ENDMDL'):
                current_model.append(line)
                models.append(''.join(current_model))
                current_model = []
                in_model = False
            elif in_model:
                current_model.append(line)

    if not models:
        # No MODEL/ENDMDL tags - treat entire file as single model
        with open(pdbqt_path) as f:
            return f.read()

    if model_index >= len(models):
        raise ValueError(f"Model {model_index} not found, file has {len(models)} models")

    return models[model_index]


def pdbqt_to_sdf(pdbqt_path: Path, sdf_path: Path, model_index: int = 0) -> bool:
    """
    Convert PDBQT to SDF.

    Args:
        pdbqt_path: Input docked PDBQT file
        sdf_path: Output SDF file
        model_index: Which binding mode to extract (0 = best)

    Returns:
        True if successful, False otherwise
    """
    obabel_bin = os.environ.get("OBABEL_BIN", "obabel")

    # Extract model
    model_content = extract_model_from_pdbqt(pdbqt_path, model_index)

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt', delete=False) as tmp:
        tmp.write(model_content)
        tmp_path = Path(tmp.name)

    try:
        # Create output directory
        sdf_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert with obabel
        cmd = [obabel_bin, str(tmp_path), "-O", str(sdf_path), "-h"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        return sdf_path.exists()

    except subprocess.CalledProcessError:
        return False

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def process_item(row: dict, model_index: int) -> dict:
    """
    Process a single PDBQT to SDF conversion.

    Args:
        row: Row from pending parquet (as dict)
        model_index: Which model to extract

    Returns:
        Result dict with compound_key, success, error
    """
    compound_key = row['compound_key']
    pdbqt_path = Path(row['docked_pdbqt_path'])
    sdf_path = Path(row['docked_sdf_path'])

    # Skip if already exists
    if sdf_path.exists():
        return {
            'compound_key': compound_key,
            'success': True,
            'skipped': True,
        }

    # Validate input exists
    if not pdbqt_path.exists():
        return {
            'compound_key': compound_key,
            'success': False,
            'error': f"Docked PDBQT not found: {pdbqt_path}",
        }

    try:
        success = pdbqt_to_sdf(pdbqt_path, sdf_path, model_index)

        return {
            'compound_key': compound_key,
            'success': success,
        }

    except Exception as e:
        return {
            'compound_key': compound_key,
            'success': False,
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
    Process a slice of conversion jobs for this worker.

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

    model_index = config.get('sdf_conversion', {}).get('select_model', 0)

    # Read slice
    df = read_slice(pending_path, task_id, num_chunks)

    if df.empty:
        print(f"Task {task_id}: Empty slice, nothing to do")
        return 0

    print(f"Task {task_id}: Processing {len(df)} conversions")

    # Process each item
    results = []
    for _, row in df.iterrows():
        result = process_item(row.to_dict(), model_index)
        results.append(result)

    # Write results
    output_path = write_results(results, results_dir, 'conversion', task_id)
    print(f"Task {task_id}: Wrote {len(results)} results to {output_path}")

    # Summary
    succeeded = sum(1 for r in results if r.get('success'))
    failed = sum(1 for r in results if not r.get('success'))
    skipped = sum(1 for r in results if r.get('skipped'))

    print(f"Task {task_id}: {succeeded} succeeded, {failed} failed, {skipped} skipped")

    return len(results)


def main():
    parser = argparse.ArgumentParser(
        description="Worker for PDBQT to SDF conversion"
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
