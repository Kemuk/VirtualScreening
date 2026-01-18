#!/usr/bin/env python3
"""
pdbqt_to_sdf.py

Convert docked PDBQT file to SDF format.

Docked PDBQT files contain multiple binding modes (models). This script
extracts a specific model and converts it to SDF for visualization and
downstream analysis (e.g., AEV-PLIG rescoring).

Default behavior: Extract the first model (best scoring pose).
"""

import argparse
import os
import sys
import subprocess
import tempfile
from pathlib import Path


def extract_model_from_pdbqt(
    pdbqt_path: Path,
    model_index: int = 0,
) -> str:
    """
    Extract a specific model from multi-model PDBQT file.

    Args:
        pdbqt_path: Path to docked PDBQT file
        model_index: Model index to extract (0 = first/best)

    Returns:
        PDBQT content for the selected model as string
    """
    if not pdbqt_path.exists():
        raise FileNotFoundError(f"PDBQT file not found: {pdbqt_path}")

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
        raise ValueError(
            f"Model index {model_index} out of range. "
            f"File contains {len(models)} models (0-{len(models)-1})"
        )

    return models[model_index]


def pdbqt_to_sdf(
    pdbqt_path: Path,
    sdf_path: Path,
    model_index: int = 0,
) -> bool:
    """
    Convert PDBQT to SDF, extracting a specific binding mode.

    Args:
        pdbqt_path: Input docked PDBQT file
        sdf_path: Output SDF file
        model_index: Which binding mode to extract (0 = best)

    Returns:
        True if successful, False otherwise
    """
    # Extract the specified model
    pdbqt_path = pdbqt_path.expanduser().resolve()
    sdf_path = sdf_path.expanduser().resolve()
    obabel_bin = os.environ.get("OBABEL_BIN", "obabel")

    try:
        model_content = extract_model_from_pdbqt(pdbqt_path, model_index)
    except Exception as e:
        print(f"ERROR: Failed to extract model {model_index}: {e}", file=sys.stderr)
        return False

    # Write model to temporary PDBQT file
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.pdbqt',
        delete=False
    ) as tmp:
        tmp.write(model_content)
        tmp_pdbqt = Path(tmp.name)

    try:
        # Convert to SDF using OpenBabel
        sdf_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            obabel_bin,
            str(tmp_pdbqt),
            "-O", str(sdf_path),
            "-h",  # Add hydrogens
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        print(f"✓ Created SDF: {sdf_path}")
        print(f"  Extracted model: {model_index}")

        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR: OpenBabel conversion failed", file=sys.stderr)
        print(f"Command: {' '.join(cmd)}", file=sys.stderr)
        print(f"Exit code: {e.returncode}", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)
        return False

    except FileNotFoundError:
        print("ERROR: obabel not found. Install OpenBabel.", file=sys.stderr)
        return False

    finally:
        # Clean up temporary file
        if tmp_pdbqt.exists():
            tmp_pdbqt.unlink()


# =============================================================================
# Batch Processing (for SLURM array jobs)
# =============================================================================

def process_batch(items: list, config: dict) -> list:
    """
    Process a batch of PDBQT to SDF conversions.

    Called by the SLURM worker to process a chunk of items.

    Args:
        items: List of item records from manifest (dicts with ligand info)
        config: Workflow configuration dict

    Returns:
        List of result records with 'ligand_id', 'success', 'error'
    """
    results = []
    model_index = config.get('sdf_conversion', {}).get('select_model', 0)

    for item in items:
        ligand_id = item['ligand_id']

        try:
            # Get paths from manifest item
            pdbqt_path = Path(item['docked_pdbqt_path'])
            sdf_path = Path(item['docked_sdf_path'])

            # Convert PDBQT to SDF
            success = pdbqt_to_sdf(
                pdbqt_path=pdbqt_path,
                sdf_path=sdf_path,
                model_index=model_index,
            )

            results.append({
                'ligand_id': ligand_id,
                'success': success,
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
        description="Convert docked PDBQT to SDF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract best pose (model 0)
  %(prog)s --input docked.pdbqt --output docked.sdf

  # Extract second pose
  %(prog)s --input docked.pdbqt --output docked.sdf --model 1

  # Batch conversion
  for f in docked_vina/actives/*_docked.pdbqt; do
      %(prog)s --input "$f" --output "docked_sdf/actives/$(basename $f .pdbqt).sdf"
  done
"""
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input docked PDBQT file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output SDF file"
    )
    parser.add_argument(
        "--model",
        type=int,
        default=0,
        help="Model index to extract (0=best, default: 0)"
    )
    parser.add_argument(
        "--ligand-id",
        type=str,
        help="Ligand identifier (for logging)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    ligand_label = args.ligand_id or args.input.stem
    print(f"Converting: {ligand_label}")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Model: {args.model}")

    # Convert PDBQT to SDF
    success = pdbqt_to_sdf(
        pdbqt_path=args.input,
        sdf_path=args.output,
        model_index=args.model,
    )

    if success:
        sys.exit(0)
    else:
        print(f"✗ Conversion failed: {ligand_label}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
