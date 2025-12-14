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
from tqdm import tqdm
import time


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
    show_progress: bool = False,
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

    # Determine vina working directory and binary name
    vina_path = Path(vina_bin)
    if vina_path.is_absolute() or '/' in vina_bin:
        # Path-like: extract directory and binary name
        vina_dir = vina_path.parent.resolve()
        vina_exec = f"./{vina_path.name}"
    else:
        # Just a command name: use current directory
        vina_dir = Path.cwd()
        vina_exec = vina_bin

    # Build Vina command with absolute paths
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
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--energy_range", str(energy_range),
        "--seed", str(seed),
    ]

    # Add mode-specific parameters
    if mode == "gpu":
        cmd.extend(["--thread", str(gpu_threads)])
    elif mode == "cpu":
        if threads is None:
            threads = os.cpu_count() or 8
        cmd.extend(["--cpu", str(threads)])
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'gpu' or 'cpu'.")

    # Prepare environment with vina directory in LD_LIBRARY_PATH
    env = os.environ.copy()
    ld_library_path = env.get('LD_LIBRARY_PATH', '')
    if ld_library_path:
        env['LD_LIBRARY_PATH'] = f"{vina_dir}:{ld_library_path}"
    else:
        env['LD_LIBRARY_PATH'] = str(vina_dir)

    # Run docking with progress bar
    try:
        if show_progress:
            # Estimate docking time for progress bar (very rough)
            estimated_time = exhaustiveness * 2  # seconds, rough estimate
            with tqdm(total=100, desc=f"Docking {ligand.name}", unit="%", ncols=80) as pbar:
                # Run in vina directory for shared library access
                result = subprocess.Popen(
                    cmd,
                    cwd=str(vina_dir),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                # Update progress bar while process runs
                start_time = time.time()
                while result.poll() is None:
                    elapsed = time.time() - start_time
                    progress = min(99, int((elapsed / estimated_time) * 100))
                    pbar.n = progress
                    pbar.refresh()
                    time.sleep(0.5)

                # Process finished
                pbar.n = 100
                pbar.refresh()

                stdout, stderr = result.communicate()

                if result.returncode != 0:
                    raise subprocess.CalledProcessError(
                        result.returncode, cmd, stdout, stderr
                    )
        else:
            # Run without progress bar
            result = subprocess.run(
                cmd,
                cwd=str(vina_dir),
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            stdout = result.stdout
            stderr = result.stderr

        # Write log file
        with open(log_file_abs, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Working directory: {vina_dir}\n")
            f.write(f"LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}\n")
            f.write(f"\nStdout:\n{stdout}\n")
            if stderr:
                f.write(f"\nStderr:\n{stderr}\n")

        print(f"✓ Docking complete: {output}")
        print(f"  Log: {log_file}")

        # Extract and display best score
        best_score = extract_best_score(stdout)
        if best_score is not None:
            print(f"  Best score: {best_score:.2f} kcal/mol")

        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Vina docking failed", file=sys.stderr)
        print(f"Command: {' '.join(cmd)}", file=sys.stderr)
        print(f"Working directory: {vina_dir}", file=sys.stderr)
        print(f"LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}", file=sys.stderr)
        print(f"Exit code: {e.returncode}", file=sys.stderr)
        print(f"Stdout: {e.stdout}", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)

        # Write error log
        with open(log_file_abs, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Working directory: {vina_dir}\n")
            f.write(f"LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}\n")
            f.write(f"\nERROR: Exit code {e.returncode}\n")
            f.write(f"\nStdout:\n{e.stdout}\n")
            f.write(f"\nStderr:\n{e.stderr}\n")

        return False

    except FileNotFoundError:
        print(f"ERROR: Vina executable not found: {vina_exec}", file=sys.stderr)
        print(f"Working directory: {vina_dir}", file=sys.stderr)
        print(f"Please install AutoDock Vina or check the tool path in config.yaml", file=sys.stderr)
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

    # Progress
    parser.add_argument("--progress", action="store_true", help="Show progress bar during docking")

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
        show_progress=args.progress,
    )

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
