#!/usr/bin/env python3
"""
aev_infer.py - Worker for AEV-PLIG neural network rescoring.

Usage:
    python -m workflow.slurm.workers.aev_infer \
        --pending data/master/pending/aev_infer.parquet \
        --task-id 0 \
        --num-chunks 500

Note: AEV-PLIG requires a specific input CSV format and runs as a
separate Python script. This worker:
1. Creates a temporary input CSV for its slice
2. Runs AEV-PLIG process_and_predict.py
3. Parses the output predictions
4. Writes results CSV
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from workflow.slurm.workers import read_slice, write_results


def prepare_aev_plig_csv(df: pd.DataFrame, output_path: Path) -> int:
    """
    Prepare AEV-PLIG input CSV from DataFrame.

    Args:
        df: DataFrame with pending items
        output_path: Path to write CSV

    Returns:
        Number of rows written
    """
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing CSV"):
        sdf_path = Path(row['docked_sdf_path'])
        pdb_path = Path(row.get('receptor_pdb_path', ''))

        if not sdf_path.exists():
            continue

        # AEV-PLIG expects: unique_id, pK, sdf_file, pdb_file
        # pK can be estimated from vina_score if available
        vina_score = row.get('vina_score')
        if pd.notna(vina_score):
            # Convert Vina score to pK: pK = -ΔG / (2.303 * R * T)
            R = 0.001987  # kcal/(mol·K)
            T = 298.0
            pK = -vina_score / (2.303 * R * T)
        else:
            pK = 0.0  # Placeholder

        rows.append({
            'unique_id': row['compound_key'],
            'pK': pK,
            'sdf_file': str(sdf_path.resolve()),
            'pdb_file': str(pdb_path.resolve()) if pdb_path.exists() else '',
        })

    if not rows:
        return 0

    csv_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(output_path, index=False)

    return len(rows)


def run_aev_plig(
    input_csv: Path,
    aev_plig_dir: Path,
    model_name: str,
    data_name: str,
) -> Path:
    """
    Run AEV-PLIG process_and_predict.py.

    Args:
        input_csv: Path to input CSV
        aev_plig_dir: Path to AEV-PLIG directory
        model_name: Name of trained model
        data_name: Name for this dataset (used in output filename)

    Returns:
        Path to predictions CSV, or None if failed
    """
    # Copy input to AEV-PLIG data directory
    aev_data_dir = aev_plig_dir / "data"
    aev_data_dir.mkdir(parents=True, exist_ok=True)

    target_csv = aev_data_dir / f"{data_name}.csv"
    import shutil
    shutil.copy(input_csv, target_csv)

    # Run AEV-PLIG
    cmd = [
        sys.executable,
        "process_and_predict.py",
        f"--dataset_csv=data/{data_name}.csv",
        f"--data_name={data_name}",
        f"--trained_model_name={model_name}",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(aev_plig_dir),
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        print(f"AEV-PLIG failed: {result.stderr}")
        return None

    # Find output file
    output_dir = aev_plig_dir / "output" / "predictions"
    possible_outputs = [
        output_dir / f"{data_name}_predictions.csv",
        output_dir / f"lit_pcba_{data_name}_predictions.csv",
    ]

    for output_path in possible_outputs:
        if output_path.exists():
            return output_path

    return None


def parse_predictions(predictions_path: Path) -> dict:
    """
    Parse AEV-PLIG predictions CSV.

    Args:
        predictions_path: Path to predictions CSV

    Returns:
        Dict mapping compound_key -> score
    """
    df = pd.read_csv(predictions_path)

    # AEV-PLIG output has unique_id and prediction columns
    # The best score is typically 'aev_plig_best_score' or similar
    score_col = None
    for col in ['aev_plig_best_score', 'prediction', 'best_score']:
        if col in df.columns:
            score_col = col
            break

    if score_col is None:
        # Use last numeric column as score
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            score_col = numeric_cols[-1]

    if score_col is None:
        return {}

    return dict(zip(df['unique_id'], df[score_col]))


def process_slice(
    pending_path: Path,
    task_id: int,
    num_chunks: int,
    config_path: Path,
    results_dir: Path,
) -> int:
    """
    Process a slice of AEV-PLIG inference jobs for this worker.

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

    aev_plig_dir = Path(config.get('tools', {}).get('aev_plig_dir', 'AEV-PLIG'))
    model_name = config.get('rescoring', {}).get('model_name', 'model_GATv2Net_ligsim90_fep_benchmark')

    # Read slice
    df = read_slice(pending_path, task_id, num_chunks)

    if df.empty:
        print(f"Task {task_id}: Empty slice, nothing to do")
        return 0

    print(f"Task {task_id}: Processing {len(df)} AEV-PLIG inferences")

    # Create temporary input CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = Path(tmpdir) / f"task_{task_id}.csv"
        data_name = f"task_{task_id}"

        num_prepared = prepare_aev_plig_csv(df, input_csv)
        if num_prepared == 0:
            print(f"Task {task_id}: No valid inputs to process")
            return 0

        print(f"Task {task_id}: Prepared {num_prepared} inputs for AEV-PLIG")

        # Run AEV-PLIG
        predictions_path = run_aev_plig(input_csv, aev_plig_dir, model_name, data_name)

        if predictions_path is None:
            print(f"Task {task_id}: AEV-PLIG failed")
            # Write all as failed with path info for debugging
            results = [
                {
                    'compound_key': row['compound_key'],
                    'ligand_id': row.get('ligand_id', ''),
                    'docked_sdf_path': row.get('docked_sdf_path', ''),
                    'success': False,
                    'score': None,
                    'error': 'AEV-PLIG failed',
                }
                for _, row in df.iterrows()
            ]
        else:
            # Parse predictions
            scores = parse_predictions(predictions_path)
            print(f"Task {task_id}: Got {len(scores)} predictions")

            # Build results with path info for debugging
            results = []
            for _, row in df.iterrows():
                compound_key = row['compound_key']
                ligand_id = row.get('ligand_id', '')
                docked_sdf_path = row.get('docked_sdf_path', '')

                if compound_key in scores:
                    results.append({
                        'compound_key': compound_key,
                        'ligand_id': ligand_id,
                        'docked_sdf_path': docked_sdf_path,
                        'success': True,
                        'score': scores[compound_key],
                        'error': '',
                    })
                else:
                    results.append({
                        'compound_key': compound_key,
                        'ligand_id': ligand_id,
                        'docked_sdf_path': docked_sdf_path,
                        'success': False,
                        'score': None,
                        'error': 'No prediction returned',
                    })

    # Write results
    output_path = write_results(results, results_dir, 'aev_infer', task_id)
    print(f"Task {task_id}: Wrote {len(results)} results to {output_path}")

    # Summary
    succeeded = sum(1 for r in results if r.get('success'))
    failed = sum(1 for r in results if not r.get('success'))

    print(f"Task {task_id}: {succeeded} succeeded, {failed} failed")

    return len(results)


def main():
    parser = argparse.ArgumentParser(
        description="Worker for AEV-PLIG neural network rescoring"
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
