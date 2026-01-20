#!/usr/bin/env python3
"""
docking.py - Worker for Vina GPU/CPU docking.

Usage:
    python -m workflow.slurm.workers.docking \
        --pending data/master/pending/docking.parquet \
        --task-id 0 \
        --num-chunks 500
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

from workflow.slurm.workers import read_slice, write_results


def extract_best_score(vina_output: str) -> float:
    """
    Extract the best binding affinity from Vina output.

    Looks for lines like:
       1        -8.5      0.000      0.000

    Returns:
        Best binding affinity (kcal/mol), or None if not found
    """
    for line in vina_output.split('\n'):
        line = line.strip()
        if line.startswith('1 ') or line.startswith('1\t'):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    pass
    return None


def run_vina_docking(
    receptor: Path,
    ligand: Path,
    output: Path,
    center_x: float,
    center_y: float,
    center_z: float,
    size_x: float,
    size_y: float,
    size_z: float,
    vina_bin: str,
    mode: str,
    exhaustiveness: int = 8,
    seed: int = 42,
    gpu_threads: int = 8000,
) -> tuple:
    """
    Run Vina docking and return (success, score).

    Returns:
        Tuple of (success: bool, score: float or None)
    """
    # Create output directory
    output.parent.mkdir(parents=True, exist_ok=True)

    # Create log directory
    log_dir = output.parent / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{ligand.stem}.log"

    # Get absolute paths
    receptor_abs = receptor.resolve()
    ligand_abs = ligand.resolve()
    output_abs = output.resolve()

    # Determine vina path
    vina_path = Path(vina_bin)
    if vina_path.is_absolute() or '/' in vina_bin:
        vina_path_abs = vina_path.resolve()
        vina_dir = vina_path_abs.parent
        vina_exec = str(vina_path_abs)
    else:
        vina_dir = Path.cwd()
        vina_exec = vina_bin

    # Build command
    cmd = [
        vina_exec,
        "--receptor", str(receptor_abs),
        "--ligand", str(ligand_abs),
        "--out", str(output_abs),
        "--center_x", str(center_x),
        "--center_y", str(center_y),
        "--center_z", str(center_z),
        "--size_x", str(size_x),
        "--size_y", str(size_y),
        "--size_z", str(size_z),
        "--seed", str(seed),
    ]

    # Mode-specific parameters
    if mode == "gpu":
        cmd.extend(["--thread", str(gpu_threads)])
    else:
        cmd.extend([
            "--exhaustiveness", str(exhaustiveness),
            "--cpu", str(os.cpu_count() or 8),
        ])

    # Run docking
    result = subprocess.run(
        cmd,
        cwd=str(vina_dir),
        capture_output=True,
        text=True,
        check=False,
    )

    # Write log
    with open(log_file, 'w') as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Exit code: {result.returncode}\n")
        f.write(f"\nStdout:\n{result.stdout}\n")
        if result.stderr:
            f.write(f"\nStderr:\n{result.stderr}\n")

    # Check success and extract score
    if output_abs.exists():
        score = extract_best_score(result.stdout)
        return True, score
    else:
        return False, None


def process_item(row: dict, config: dict, targets_config: dict) -> dict:
    """
    Process a single docking job.

    Args:
        row: Row from pending parquet (as dict)
        config: Workflow config
        targets_config: Targets config with box parameters

    Returns:
        Result dict with compound_key, ligand_id, paths, success, score, error
    """
    compound_key = row['compound_key']
    ligand_id = row.get('ligand_id', '')
    protein_id = row['protein_id']

    # Get paths (store as strings for CSV output)
    ligand_pdbqt_path = row.get('ligand_pdbqt_path', '')
    docked_pdbqt_path = row.get('docked_pdbqt_path', '')

    ligand_path = Path(ligand_pdbqt_path) if ligand_pdbqt_path else None
    receptor_path = Path(row['receptor_pdbqt_path']) if row.get('receptor_pdbqt_path') else None
    output_path = Path(docked_pdbqt_path) if docked_pdbqt_path else None

    # Compute expected log path (output_dir/log/ligand_stem.log)
    if output_path and ligand_path:
        docking_log_path = str(output_path.parent / "log" / f"{ligand_path.stem}.log")
    else:
        docking_log_path = ''

    # Base result with identifying info and paths
    base_result = {
        'compound_key': compound_key,
        'ligand_id': ligand_id,
        'ligand_pdbqt_path': ligand_pdbqt_path,
        'docked_pdbqt_path': docked_pdbqt_path,
        'docking_log_path': docking_log_path,
    }

    # Skip if already exists
    if output_path and output_path.exists():
        # Try to extract score from existing file
        try:
            with open(output_path) as f:
                content = f.read()
            score = extract_best_score(content)
        except Exception:
            score = None

        return {
            **base_result,
            'success': True,
            'skipped': True,
            'score': score,
            'error': '',
        }

    # Validate inputs exist
    if not ligand_path or not ligand_path.exists():
        return {
            **base_result,
            'success': False,
            'skipped': False,
            'score': None,
            'error': f"Ligand not found: {ligand_pdbqt_path}",
        }

    if not receptor_path or not receptor_path.exists():
        return {
            **base_result,
            'success': False,
            'skipped': False,
            'score': None,
            'error': f"Receptor not found: {row.get('receptor_pdbqt_path', '')}",
        }

    try:
        # Get box parameters from targets config
        target_cfg = targets_config['targets'][protein_id]
        box_center = target_cfg['box_center']
        box_size = target_cfg.get('box_size', config.get('default_box_size', {}))

        # Get docking config
        docking_config = config.get('docking', {})
        mode = docking_config.get('mode', 'cpu')
        tools = config.get('tools', {})

        if mode == 'gpu':
            vina_bin = tools.get('vina_gpu', 'vina')
        else:
            vina_bin = tools.get('vina_cpu', 'vina')

        # Run docking
        success, score = run_vina_docking(
            receptor=receptor_path,
            ligand=ligand_path,
            output=output_path,
            center_x=box_center['x'],
            center_y=box_center['y'],
            center_z=box_center['z'],
            size_x=box_size.get('x', 25.0),
            size_y=box_size.get('y', 25.0),
            size_z=box_size.get('z', 25.0),
            vina_bin=vina_bin,
            mode=mode,
            exhaustiveness=docking_config.get('exhaustiveness', 8),
            seed=docking_config.get('seed', 42),
            gpu_threads=config.get('gpu', {}).get('threads', 8000),
        )

        return {
            **base_result,
            'success': success,
            'skipped': False,
            'score': score,
            'error': '' if success else 'Docking failed - no output file created',
        }

    except Exception as e:
        return {
            **base_result,
            'success': False,
            'skipped': False,
            'score': None,
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
    Process a slice of docking jobs for this worker.

    Args:
        pending_path: Path to pending parquet
        task_id: SLURM_ARRAY_TASK_ID
        num_chunks: Total number of chunks
        config_path: Path to config.yaml
        results_dir: Directory for result CSV files

    Returns:
        Number of items processed
    """
    # Load configs
    with open(config_path) as f:
        config = yaml.safe_load(f)

    targets_path = Path(config.get('targets_config', 'config/targets.yaml'))
    with open(targets_path) as f:
        targets_config = yaml.safe_load(f)

    # Read slice
    df = read_slice(pending_path, task_id, num_chunks)

    if df.empty:
        print(f"Task {task_id}: Empty slice, nothing to do")
        return 0

    print(f"Task {task_id}: Processing {len(df)} docking jobs")

    # Process each item with progress bar
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Task {task_id}"):
        result = process_item(row.to_dict(), config, targets_config)
        results.append(result)

    # Write results
    output_path = write_results(results, results_dir, 'docking', task_id)
    print(f"Task {task_id}: Wrote {len(results)} results to {output_path}")

    # Summary
    succeeded = sum(1 for r in results if r.get('success'))
    failed = sum(1 for r in results if not r.get('success'))
    skipped = sum(1 for r in results if r.get('skipped'))
    with_score = sum(1 for r in results if r.get('score') is not None)

    print(f"Task {task_id}: {succeeded} succeeded, {failed} failed, {skipped} skipped, {with_score} with scores")

    return len(results)


def main():
    parser = argparse.ArgumentParser(
        description="Worker for Vina GPU/CPU docking"
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
