#!/usr/bin/env python3
"""
scan_manifest_items.py

Lightweight scanning of SMILES files to generate manifest items.
This script does NOT use RDKit - it just reads files and counts ligands.

Used by the orchestrator to prepare items for chunking before
submitting array jobs for parallel manifest creation.

Usage:
    python workflow/scripts/scan_manifest_items.py --config config/config.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def load_config(config_path: Path) -> Dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def count_smiles_file(smiles_path: Path) -> int:
    """
    Count valid lines in a SMILES file (fast, no parsing).

    Args:
        smiles_path: Path to SMILES file

    Returns:
        Number of valid ligand lines
    """
    if not smiles_path.exists():
        return 0

    count = 0
    with open(smiles_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    count += 1
    return count


def parse_smiles_file_lazy(smiles_path: Path) -> List[Tuple[str, str]]:
    """
    Parse SMILES file and return list of (ligand_id, smiles).
    Expects format: SMILES [whitespace] ligand_id

    This is a lightweight parser - no RDKit validation.
    """
    ligands = []
    if not smiles_path.exists():
        return ligands

    with open(smiles_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) >= 2:
                smiles = parts[0]
                ligand_id = parts[1]
                ligands.append((ligand_id, smiles))

    return ligands


def scan_targets(
    targets_config: Dict,
    workflow_config: Dict,
    project_root: Path,
    max_items: int = None,
) -> List[Dict]:
    """
    Scan all targets and generate lightweight item records.

    This does NOT process with RDKit - just reads SMILES files and
    creates item records that can be chunked for parallel processing.

    Args:
        targets_config: Targets configuration dict
        workflow_config: Workflow configuration dict
        project_root: Project root path
        max_items: Maximum total items to return (for devel mode). None = no limit.

    Returns:
        List of item dicts ready for chunking
    """
    dataset_name = workflow_config.get('dataset', 'LIT_PCBA')
    default_box_size = workflow_config.get('default_box_size', {'x': 25.0, 'y': 25.0, 'z': 25.0})
    targets = targets_config.get('targets', {})

    # Get mode and limits
    mode = workflow_config.get('mode', 'test')
    if mode == 'test':
        max_actives = workflow_config.get('test', {}).get('actives_per_protein', 100)
        max_inactives = workflow_config.get('test', {}).get('inactives_per_protein', 9900)
    else:
        max_actives = None
        max_inactives = None

    items = []

    for protein_id, target_config in targets.items():
        # Early exit if we've hit max_items
        if max_items is not None and len(items) >= max_items:
            break
        receptor_mol2 = project_root / target_config['receptor_mol2']
        actives_smi = project_root / target_config['actives_smi']
        inactives_smi = project_root / target_config['inactives_smi']

        box_center = target_config['box_center']
        box_size = target_config.get('box_size', default_box_size)

        target_dir = str(receptor_mol2.parent.relative_to(project_root))

        # Process actives
        actives_count = 0
        for ligand_id, smiles in parse_smiles_file_lazy(actives_smi):
            if max_actives is not None and actives_count >= max_actives:
                break
            if max_items is not None and len(items) >= max_items:
                break

            items.append({
                'ligand_id': ligand_id,
                'protein_id': protein_id,
                'dataset': dataset_name,
                'smiles': smiles,
                'is_active': True,
                'source_smiles_file': str(actives_smi.relative_to(project_root)),
                'receptor_mol2': str(receptor_mol2.relative_to(project_root)),
                'target_dir': target_dir,
                'box_center_x': float(box_center['x']),
                'box_center_y': float(box_center['y']),
                'box_center_z': float(box_center['z']),
                'box_size_x': float(box_size['x']),
                'box_size_y': float(box_size['y']),
                'box_size_z': float(box_size['z']),
            })
            actives_count += 1

        # Process inactives
        inactives_count = 0
        for ligand_id, smiles in parse_smiles_file_lazy(inactives_smi):
            if max_inactives is not None and inactives_count >= max_inactives:
                break
            if max_items is not None and len(items) >= max_items:
                break

            items.append({
                'ligand_id': ligand_id,
                'protein_id': protein_id,
                'dataset': dataset_name,
                'smiles': smiles,
                'is_active': False,
                'source_smiles_file': str(inactives_smi.relative_to(project_root)),
                'receptor_mol2': str(receptor_mol2.relative_to(project_root)),
                'target_dir': target_dir,
                'box_center_x': float(box_center['x']),
                'box_center_y': float(box_center['y']),
                'box_center_z': float(box_center['z']),
                'box_size_x': float(box_size['x']),
                'box_size_y': float(box_size['y']),
                'box_size_z': float(box_size['z']),
            })
            inactives_count += 1

        print(f"  {protein_id}: {actives_count} actives + {inactives_count} inactives")

    return items


def main():
    parser = argparse.ArgumentParser(
        description="Scan SMILES files to generate manifest items for chunking"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to main config file",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("config/targets.yaml"),
        help="Path to targets config file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for items (default: stdout)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only count items, don't output full list",
    )

    args = parser.parse_args()

    # Load configurations
    workflow_config = load_config(args.config)
    targets_config = load_config(args.targets)

    print("Scanning SMILES files...")
    items = scan_targets(
        targets_config=targets_config,
        workflow_config=workflow_config,
        project_root=args.project_root,
    )

    print(f"\nTotal items: {len(items)}")

    if args.count_only:
        return 0

    # Output items
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(items, f, indent=2)
        print(f"Wrote items to: {args.output}")
    else:
        # Write to stdout as JSON
        json.dump(items, sys.stdout, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
