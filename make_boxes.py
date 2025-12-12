#!/usr/bin/env python3
"""
make_boxes.py

Generate docking box configurations for each protein target in LIT_PCBA.
Reproduces GNINA’s LIT-PCBA docking setup:
  --seed 0 --autobox_add 16 --num_modes 20 (exhaustiveness=8 default)

Outputs: vina_boxes.csv with one row per target directory that contains *.mol2 files.
Now also records config files and paths to actives.smi/inactives.smi files.
"""

import csv
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from rdkit import Chem

# --------------------- GNINA defaults ---------------------
PAD = 16.0
MIN_SIZE = 18.0
MAX_SIZE = 36.0
THREAD = int(os.getenv("THREAD", 8000))
SEED = 0
NUM_MODES = 9
EXHAUSTIVENESS = 8

DATA_DIR = Path(os.getenv("DATA_DIR", "./LIT_PCBA")).resolve()
MASTER_CSV = DATA_DIR / "vina_boxes.csv"

# --------------------- Box Calculation ---------------------
def compute_box_from_mol2(protein_root: Path):
    """Compute GNINA-style box using *_ligand.mol2 if available,
    else fallback to *_protein.mol2 centroid + 22 Å cube."""
    ligand_files = list(protein_root.glob("*_ligand.mol2"))
    receptor_files = list(protein_root.glob("*_protein.mol2"))

    ligand_file = ligand_files[0] if ligand_files else None
    receptor_file = receptor_files[0] if receptor_files else None

    # --- Use ligand if present ---
    if ligand_file and ligand_file.exists():
        mol = Chem.MolFromMol2File(str(ligand_file), removeHs=False)
        if mol is not None and mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            xs, ys, zs = [], [], []
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 1:  # skip hydrogens
                    continue
                pos = conf.GetAtomPosition(atom.GetIdx())
                xs.append(pos.x); ys.append(pos.y); zs.append(pos.z)
            if xs:
                cx, cy, cz = (
                    (max(xs) + min(xs)) / 2,
                    (max(ys) + min(ys)) / 2,
                    (max(zs) + min(zs)) / 2,
                )
                clamp = lambda v: max(MIN_SIZE, min(MAX_SIZE, v))
                sx = clamp((max(xs) - min(xs)) + 2 * PAD)
                sy = clamp((max(ys) - min(ys)) + 2 * PAD)
                sz = clamp((max(zs) - min(zs)) + 2 * PAD)
                return cx, cy, cz, sx, sy, sz

    # --- Fallback: receptor centroid ---
    if receptor_file and receptor_file.exists():
        mol = Chem.MolFromMol2File(str(receptor_file), removeHs=False)
        if mol is not None and mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            xs, ys, zs = [], [], []
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 1:
                    continue
                pos = conf.GetAtomPosition(atom.GetIdx())
                xs.append(pos.x); ys.append(pos.y); zs.append(pos.z)
            if xs:
                cx, cy, cz = (
                    sum(xs) / len(xs),
                    sum(ys) / len(ys),
                    sum(zs) / len(zs),
                )
                return cx, cy, cz, 22.0, 22.0, 22.0

    raise RuntimeError(f"No usable ligand or protein mol2 found in {protein_root}")

# --------------------- Worker ---------------------
def process_target(protein_root: Path) -> dict:
    target = protein_root.name
    receptor_pdbqt = next(protein_root.glob("*_protein.pdbqt"), None)

    actives_dir = protein_root / "pdbqt/actives"
    inactives_dir = protein_root / "pdbqt/inactives"
    actives_dir.mkdir(parents=True, exist_ok=True)
    inactives_dir.mkdir(parents=True, exist_ok=True)

    cx, cy, cz, sx, sy, sz = compute_box_from_mol2(protein_root)

    docked_vina_actives = protein_root / "docked_vina/actives"
    docked_vina_inactives = protein_root / "docked_vina/inactives"
    log_actives = docked_vina_actives / "log"
    log_inactives = docked_vina_inactives / "log"
    docked_sdf_actives = protein_root / "docked_sdf/actives"
    docked_sdf_inactives = protein_root / "docked_sdf/inactives"
    out_dir = protein_root / "pdbqt/out"
    aev_plig_csv = protein_root / f"rescoring/datasets/aev_plig_{target}.csv"

    # Config files if present
    config_actives = next(protein_root.glob("*config_actives.txt"), None)
    config_inactives = next(protein_root.glob("*config_inactives.txt"), None)

    # --- NEW SMILE FILES ---
    actives_smi_file = protein_root / "actives.smi"
    inactives_smi_file = protein_root / "inactives.smi"

    for p in [
        docked_vina_actives, docked_vina_inactives, log_actives, log_inactives,
        docked_sdf_actives, docked_sdf_inactives, out_dir, aev_plig_csv.parent,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    return dict(
        target=target,
        receptor_pdbqt=str(receptor_pdbqt) if receptor_pdbqt else "",
        actives_dir=str(actives_dir),
        inactives_dir=str(inactives_dir),
        docked_vina_actives=str(docked_vina_actives),
        docked_vina_inactives=str(docked_vina_inactives),
        log_actives=str(log_actives),
        log_inactives=str(log_inactives),
        docked_sdf_actives=str(docked_sdf_actives),
        docked_sdf_inactives=str(docked_sdf_inactives),
        out_dir=str(out_dir),
        aev_plig_csv=str(aev_plig_csv),
        center_x=cx, center_y=cy, center_z=cz,
        size_x=sx, size_y=sy, size_z=sz,
        thread=THREAD,
        seed=SEED,
        num_modes=NUM_MODES,
        exhaustiveness=EXHAUSTIVENESS,
        config_actives=str(config_actives) if config_actives else "",
        config_inactives=str(config_inactives) if config_inactives else "",
        # --- ADDED NEW COLUMNS ---
        actives_smile=str(actives_smi_file),
        inactives_smile=str(inactives_smi_file),
    )

# --------------------- Main ---------------------
def main():
    protein_roots = [
        d for d in DATA_DIR.iterdir()
        if d.is_dir() and any(d.glob("*.mol2"))
    ]
    print(f"Found {len(protein_roots)} targets with *.mol2 files")

    results = []
    with ProcessPoolExecutor() as ex:
        futures = {ex.submit(process_target, root): root for root in protein_roots}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing targets"):
            try:
                results.append(fut.result())
            except Exception as e:
                root_name = futures[fut].name
                print(f"ERROR processing target {root_name}: {e}")
                
    if not results:
        print("No targets were successfully processed. Exiting.")
        return

    # Sort results by target name for consistent CSV output
    results.sort(key=lambda r: r['target'])

    with MASTER_CSV.open("w", newline="") as f:
        # Fieldnames are taken from the first result's keys
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote master CSV: {MASTER_CSV}")

if __name__ == "__main__":
    main()