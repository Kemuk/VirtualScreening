#!/usr/bin/env python3
"""
update_box_centers.py

Compute docking box centers from crystal ligands and update targets.yaml.
Mimics GNINA's --autobox_ligand approach: center is the centroid of
heavy atoms (non-hydrogens) in the crystal ligand.

Usage:
    python workflow/scripts/update_box_centers.py
    python workflow/scripts/update_box_centers.py --dry-run
    python workflow/scripts/update_box_centers.py --data-dir /path/to/LIT_PCBA
"""

import argparse
from pathlib import Path

import yaml
from rdkit import Chem


def compute_center_from_mol2(ligand_path: Path) -> tuple[float, float, float]:
    """
    Compute the centroid of heavy atoms in a MOL2 ligand file.

    This mimics GNINA's --autobox_ligand behavior where the box center
    is placed at the centroid of the reference ligand.

    Args:
        ligand_path: Path to the crystal ligand MOL2 file

    Returns:
        Tuple of (center_x, center_y, center_z) coordinates

    Raises:
        ValueError: If ligand cannot be parsed or has no heavy atoms
    """
    mol = Chem.MolFromMol2File(str(ligand_path), removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to parse ligand: {ligand_path}")

    if mol.GetNumConformers() == 0:
        raise ValueError(f"Ligand has no conformers: {ligand_path}")

    conf = mol.GetConformer()
    xs, ys, zs = [], [], []

    for atom in mol.GetAtoms():
        # Skip hydrogens
        if atom.GetAtomicNum() == 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        xs.append(pos.x)
        ys.append(pos.y)
        zs.append(pos.z)

    if not xs:
        raise ValueError(f"No heavy atoms found in ligand: {ligand_path}")

    # Compute centroid (same as GNINA autobox)
    center_x = (max(xs) + min(xs)) / 2
    center_y = (max(ys) + min(ys)) / 2
    center_z = (max(zs) + min(zs)) / 2

    return center_x, center_y, center_z


def find_crystal_ligand(data_dir: Path, target: str) -> Path | None:
    """
    Find the crystal ligand MOL2 file for a target.

    Looks for: {data_dir}/{target}/{target}_ligand.mol2
    Also tries: {data_dir}/{target}/*_ligand.mol2

    Args:
        data_dir: Path to LIT_PCBA data directory
        target: Target name (e.g., "ADRB2")

    Returns:
        Path to ligand file if found, None otherwise
    """
    target_dir = data_dir / target
    if not target_dir.exists():
        return None

    # Try exact match first
    exact_path = target_dir / f"{target}_ligand.mol2"
    if exact_path.exists():
        return exact_path

    # Fallback: any *_ligand.mol2
    ligand_files = list(target_dir.glob("*_ligand.mol2"))
    if ligand_files:
        return ligand_files[0]

    return None


def load_targets_yaml(yaml_path: Path) -> dict:
    """Load targets.yaml file."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def save_targets_yaml(yaml_path: Path, data: dict) -> None:
    """Save targets.yaml file, preserving structure."""
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def main():
    parser = argparse.ArgumentParser(
        description="Update box_center values in targets.yaml from crystal ligands"
    )
    parser.add_argument(
        "--targets-yaml",
        type=Path,
        default=Path("config/targets.yaml"),
        help="Path to targets.yaml (default: config/targets.yaml)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./LIT_PCBA"),
        help="Path to LIT_PCBA data directory (default: ./LIT_PCBA)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print computed values without modifying targets.yaml",
    )
    args = parser.parse_args()

    # Resolve paths
    targets_yaml = args.targets_yaml.resolve()
    data_dir = args.data_dir.resolve()

    if not targets_yaml.exists():
        print(f"Error: targets.yaml not found: {targets_yaml}")
        return 1

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1

    # Load current configuration
    config = load_targets_yaml(targets_yaml)
    targets = config.get("targets", {})

    if not targets:
        print("Error: No targets found in targets.yaml")
        return 1

    print(f"Found {len(targets)} targets in {targets_yaml}")
    print(f"Looking for crystal ligands in {data_dir}")
    print()

    updated = 0
    skipped = 0
    errors = 0

    for target_name, target_config in targets.items():
        ligand_path = find_crystal_ligand(data_dir, target_name)

        if ligand_path is None:
            print(f"  {target_name}: No crystal ligand found, skipping")
            skipped += 1
            continue

        try:
            cx, cy, cz = compute_center_from_mol2(ligand_path)

            # Round to 1 decimal place for cleaner YAML
            cx, cy, cz = round(cx, 1), round(cy, 1), round(cz, 1)

            old_center = target_config.get("box_center", {})
            old_str = f"({old_center.get('x', 0)}, {old_center.get('y', 0)}, {old_center.get('z', 0)})"
            new_str = f"({cx}, {cy}, {cz})"

            print(f"  {target_name}: {old_str} -> {new_str}")

            if not args.dry_run:
                target_config["box_center"] = {"x": cx, "y": cy, "z": cz}

            updated += 1

        except Exception as e:
            print(f"  {target_name}: Error - {e}")
            errors += 1

    print()
    print(f"Summary: {updated} updated, {skipped} skipped, {errors} errors")

    if not args.dry_run and updated > 0:
        save_targets_yaml(targets_yaml, config)
        print(f"Wrote updated configuration to {targets_yaml}")
    elif args.dry_run:
        print("(dry-run mode, no changes written)")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    exit(main())
