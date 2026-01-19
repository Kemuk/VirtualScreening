#!/usr/bin/env python3
"""
dock_vina.py

Perform molecular docking using AutoDock Vina (GPU or CPU).

This script docks a single ligand to a receptor. Batch processing
is handled by Snakemake parallel execution.

Output:
  - Docked PDBQT file with multiple binding modes
  - Log file with binding affinities and RMSD values
"""

import argparse
import sys
import subprocess
import os
from pathlib import Path


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
    vina_bin: str = "vina",
    exhaustiveness: int = 8,
    num_modes: int = 9,
    energy_range: int = 3,
    seed: int = 42,
    threads: int = None,
    gpu_threads: int = 8000,
    mode: str = "cpu",
) -> bool:
    """
    Run AutoDock Vina docking.

    Args:
        receptor: Receptor PDBQT file
        ligand: Ligand PDBQT file
        output: Output docked PDBQT file
        center_x, center_y, center_z: Box center coordinates
        size_x, size_y, size_z: Box dimensions
        vina_bin: Path to Vina executable
        exhaustiveness: Search exhaustiveness
        num_modes: Number of binding modes
        energy_range: Energy range for modes (kcal/mol)
        seed: Random seed
        threads: Number of CPU threads (for CPU mode)
        gpu_threads: Number of GPU threads (for GPU mode)
        mode: 'gpu' or 'cpu'

    Returns:
        True if successful, False otherwise
    """
    # Create output directory
    output.parent.mkdir(parents=True, exist_ok=True)

    # Create log directory
    log_dir = output.parent / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{ligand.stem}.log"

    # Get absolute paths (needed for changing working directory)
    receptor_abs = receptor.resolve()
    ligand_abs = ligand.resolve()
    output_abs = output.resolve()
    log_file_abs = log_file.resolve()

    # Determine vina working directory and binary absolute path
    vina_path = Path(vina_bin)
    if vina_path.is_absolute() or '/' in vina_bin:
        # Path-like: use absolute path to executable
        vina_path_abs = vina_path.resolve()
        vina_dir = vina_path_abs.parent
        vina_exec = str(vina_path_abs)
    else:
        # Just a command name: use current directory
        vina_dir = Path.cwd()
        vina_exec = vina_bin

    # Build Vina command with common parameters (absolute paths)
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

    # Add mode-specific parameters
    if mode == "gpu":
        # GPU mode: no exhaustiveness or energy_range
        cmd.extend(["--thread", str(gpu_threads)])
    elif mode == "cpu":
        # CPU mode: includes exhaustiveness and energy_range
        cmd.extend([
            "--exhaustiveness", str(exhaustiveness),
            "--energy_range", str(energy_range),
        ])
        if threads is None:
            threads = os.cpu_count() or 8
        cmd.extend(["--cpu", str(threads)])
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'gpu' or 'cpu'.")

    # Run docking (use check=False like legacy)
    result = subprocess.run(
        cmd,
        cwd=str(vina_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = result.stdout
    stderr = result.stderr
    returncode = result.returncode

    # Write log file
    with open(log_file_abs, 'w') as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Working directory: {vina_dir}\n")
        f.write(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')}\n")
        f.write(f"Exit code: {returncode}\n")
        f.write(f"\nStdout:\n{stdout}\n")
        if stderr:
            f.write(f"\nStderr:\n{stderr}\n")

    # Determine success by checking if output file exists (like legacy)
    if output_abs.exists():
        print(f"✓ Docking complete: {output}")
        print(f"  Log: {log_file}")

        # Extract and display best score
        best_score = extract_best_score(stdout)
        if best_score is not None:
            print(f"  Best score: {best_score:.2f} kcal/mol")

        return True
    else:
        print(f"ERROR: Vina docking failed - output file not created", file=sys.stderr)
        print(f"Command: {' '.join(cmd)}", file=sys.stderr)
        print(f"Working directory: {vina_dir}", file=sys.stderr)
        print(f"Exit code: {returncode}", file=sys.stderr)
        print(f"Stdout: {stdout}", file=sys.stderr)
        print(f"Stderr: {stderr}", file=sys.stderr)
        return False


def extract_best_score(vina_output: str) -> float:
    """
    Extract the best (first) binding affinity from Vina output.

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


# =============================================================================
# Batch Processing (for SLURM array jobs)
# =============================================================================

def process_batch(items: list, config: dict) -> list:
    """
    Process a batch of docking jobs.

    Called by the SLURM worker to process a chunk of items.

    Args:
        items: List of item records from manifest (dicts with ligand info)
        config: Workflow configuration dict

    Returns:
        List of result records with 'ligand_id', 'success', 'score', 'error'
    """
    import yaml
    from pathlib import Path

    results = []
    docking_config = config.get('docking', {})
    tools = config.get('tools', {})

    # Determine vina binary based on mode
    mode = docking_config.get('mode', 'cpu')
    if mode == 'gpu':
        vina_bin = tools.get('vina_gpu', 'vina')
    else:
        vina_bin = tools.get('vina_cpu', 'vina')

    # Load targets config for box parameters
    targets_path = Path(config.get('targets_config', 'config/targets.yaml'))
    with open(targets_path) as f:
        targets_config = yaml.safe_load(f)

    for item in items:
        ligand_id = item['ligand_id']
        protein_id = item['protein_id']

        try:
            # Get paths from manifest item
            ligand_path = Path(item['ligand_pdbqt_path'])
            receptor_path = Path(item['receptor_pdbqt_path'])
            output_path = Path(item['docked_pdbqt_path'])

            # Get box parameters from targets config
            target_cfg = targets_config['targets'][protein_id]
            box_center = target_cfg['box_center']
            box_size = target_cfg.get('box_size', config.get('default_box_size', {}))

            # Run docking
            success = run_vina_docking(
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
                exhaustiveness=docking_config.get('exhaustiveness', 8),
                num_modes=docking_config.get('num_modes', 9),
                energy_range=docking_config.get('energy_range', 3),
                seed=docking_config.get('seed', 42),
                mode=mode,
            )

            # Extract score from output if successful
            score = None
            if success and output_path.exists():
                with open(output_path) as f:
                    content = f.read()
                    score = extract_best_score(content)

            results.append({
                'ligand_id': ligand_id,
                'success': success,
                'score': score,
            })

        except Exception as e:
            results.append({
                'ligand_id': ligand_id,
                'success': False,
                'error': str(e),
            })

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Perform molecular docking using AutoDock Vina"
    )

    # Input/output
    parser.add_argument("--receptor", type=Path, required=True, help="Receptor PDBQT file")
    parser.add_argument("--ligand", type=Path, required=True, help="Ligand PDBQT file")
    parser.add_argument("--output", type=Path, required=True, help="Output docked PDBQT file")

    # Box parameters
    parser.add_argument("--center-x", type=float, required=True, help="Box center X")
    parser.add_argument("--center-y", type=float, required=True, help="Box center Y")
    parser.add_argument("--center-z", type=float, required=True, help="Box center Z")
    parser.add_argument("--size-x", type=float, required=True, help="Box size X")
    parser.add_argument("--size-y", type=float, required=True, help="Box size Y")
    parser.add_argument("--size-z", type=float, required=True, help="Box size Z")

    # Vina parameters
    parser.add_argument("--vina-bin", type=str, default="vina", help="Path to Vina executable")
    parser.add_argument("--exhaustiveness", type=int, default=8, help="Search exhaustiveness")
    parser.add_argument("--num-modes", type=int, default=9, help="Number of binding modes")
    parser.add_argument("--energy-range", type=int, default=3, help="Energy range (kcal/mol)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Mode
    parser.add_argument("--mode", choices=["gpu", "cpu"], default="cpu", help="Execution mode")
    parser.add_argument("--threads", type=int, help="CPU threads (CPU mode only)")
    parser.add_argument("--gpu-threads", type=int, default=8000, help="GPU threads (GPU mode only)")

    args = parser.parse_args()

    # Validate inputs
    if not args.receptor.exists():
        print(f"ERROR: Receptor file not found: {args.receptor}", file=sys.stderr)
        sys.exit(1)

    if not args.ligand.exists():
        print(f"ERROR: Ligand file not found: {args.ligand}", file=sys.stderr)
        sys.exit(1)

    # Log parameters
    print(f"Docking: {args.ligand.name}")
    print(f"  Receptor: {args.receptor}")
    print(f"  Box center: ({args.center_x:.1f}, {args.center_y:.1f}, {args.center_z:.1f})")
    print(f"  Box size: ({args.size_x:.1f}, {args.size_y:.1f}, {args.size_z:.1f})")
    print(f"  Mode: {args.mode}")
    print(f"  Exhaustiveness: {args.exhaustiveness}")
    print(f"  Num modes: {args.num_modes}")

    # Run docking
    success = run_vina_docking(
        receptor=args.receptor,
        ligand=args.ligand,
        output=args.output,
        center_x=args.center_x,
        center_y=args.center_y,
        center_z=args.center_z,
        size_x=args.size_x,
        size_y=args.size_y,
        size_z=args.size_z,
        vina_bin=args.vina_bin,
        exhaustiveness=args.exhaustiveness,
        num_modes=args.num_modes,
        energy_range=args.energy_range,
        seed=args.seed,
        threads=args.threads,
        gpu_threads=args.gpu_threads,
        mode=args.mode,
    )

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
