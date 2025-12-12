#!/usr/bin/env python3
"""
pdbqt2sdf.py

- Default boxes CSV: LIT_PCBA/vina_boxes.csv
- Prepare master manifest (multithreaded) from that CSV.
- Read the master manifest and run conversions for rows assigned to this shard.
- Converts .pdbqt -> .sdf using RDKit and a SMILES template string.
- The SMILES string is sourced from the `actives_smile` / `inactives_smile` files.
- Selects the first MODEL (conformer 0) from the PDBQT for coordinates.
- Writes per-ligand debug logs to <docked_sdf_dir>/logs/<ligand>.log
- Does NOT modify the master manifest after running.
"""
from pathlib import Path
import argparse
import csv
import os
import subprocess
import time
import multiprocessing
import functools   # <-- IMPORTED
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.contrib.concurrent import thread_map as tmap
from tqdm import tqdm
import random
import numpy as np
import warnings

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolfiles  # <-- MODIFIED IMPORT

# Suppress RDKit warnings
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')
warnings.filterwarnings("ignore", category=UserWarning)


# deterministic seeds
os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)
random.seed(42)

# Updated required columns
REQUIRED_COLS = [
    "target","receptor_pdbqt","actives_dir","inactives_dir",
    "docked_sdf_actives","docked_sdf_inactives",
    "actives_smile", "inactives_smile" # <-- UPDATED
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="LIT_PCBA/vina_boxes.csv", help="Boxes CSV (default LIT_PCBA/vina_boxes.csv)")
    p.add_argument("--prepare_master", action="store_true")
    p.add_argument("--master_manifest", default="LIT_PCBA/sdf_manifest.csv")
    p.add_argument("--shard_total", type=int, default=100)
    p.add_argument("--task_id", type=int, default=-1, help="If <0 use SLURM_ARRAY_TASK_ID or 0")
    p.add_argument("--obabel_bin", default="obabel", help="(Ignored) kept for CLI compatibility")
    p.add_argument("--test_run", action="store_true", help="Limit to first ligand per dir")
    p.add_argument("--prepare_workers", type=int, default=0)
    p.add_argument("--runtime_workers", type=int, default=0)
    return p.parse_args()

def _read_numpy_csv(csv_path: Path):
    a = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if a.shape == (): a = np.array([a], dtype=a.dtype)
    return a

@functools.lru_cache(maxsize=None)  # <-- OPTIMIZATION 1: Cache SMILES file reads
def smi_file_to_dict(smi_file_path: Path) -> dict[str, str]:
    """Reads a SMI file (SMILES <ID>) and returns a {ID: SMILES} map."""
    smi_map = {}
    if not smi_file_path.exists():
        print(f"Warning: SMILES file not found: {smi_file_path}")
        return smi_map
    with open(smi_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                smi, name = parts[0], " ".join(parts[1:]) # Handle names with spaces
                smi_map[name] = smi
    return smi_map

def collect_ligand_entries(boxes_csv: Path, test_run: bool = False, workers: int = 0):
    """
    From boxes CSV produce list of entries:
    (target, receptor, ligpath, template_smi_string, outpath)
    
    OPTIMIZED: Globs all unique directories in parallel first to avoid I/O
    contention in the main processing loop.
    """
    import csv as _csv
    with open(boxes_csv, newline="") as f:
        rdr = _csv.reader(f); header = next(rdr)
    missing = [c for c in REQUIRED_COLS if c not in header]
    if missing:
        raise SystemExit(f"Missing columns in boxes CSV: {missing}")
    arr = _read_numpy_csv(boxes_csv)
    targets = arr["target"]
    receptors = [str(Path(p).resolve()) for p in arr["receptor_pdbqt"]]
    
    # Docked PDBQT input dirs
    act_in = [Path(p) for p in arr["actives_dir"]]
    inact_in = [Path(p) for p in arr["inactives_dir"]]
    
    # SDF output dirs
    act_out = [Path(p) for p in arr["docked_sdf_actives"]]
    inact_out = [Path(p) for p in arr["docked_sdf_inactives"]]
    
    # Template SMI input *files*
    smi_act_in = [Path(p) for p in arr["actives_smile"]]
    smi_inact_in = [Path(p) for p in arr["inactives_smile"]]

    n_boxes = arr.shape[0]
    if workers <= 0:
        workers = min(32, max(1, multiprocessing.cpu_count()))

    # --- START OPTIMIZATION ---
    # 1. Get all unique input directories
    all_act_dirs = set(act_in)
    all_inact_dirs = set(inact_in)
    all_unique_dirs = list(all_act_dirs.union(all_inact_dirs))
    
    print(f"Found {len(all_unique_dirs)} unique PDBQT directories to scan.")

    # 2. Define a parallel globbing helper
    def glob_pdbqt(d: Path):
        """Globs a directory and returns its path and sorted list of PDBQTs."""
        if not d.is_dir():
            return (d, [])
        return (d, sorted(d.glob("*.pdbqt")))

    # 3. Run the globbing in parallel
    glob_results = tmap(
        glob_pdbqt, 
        all_unique_dirs, 
        max_workers=workers, 
        desc="Scanning all input dirs"
    )
    
    # 4. Create a fast lookup map: {Path: [list_of_files]}
    master_glob_map = dict(glob_results)
    
    # --- END OPTIMIZATION ---

    def process_box(i):
        local = []
        a_in, a_out = act_in[i], act_out[i]
        ia_in, ia_out = inact_in[i], inact_out[i]
        
        # Load the SMILES files into lookup maps (fast, cached)
        smi_a_file, smi_ia_file = smi_act_in[i], smi_inact_in[i]
        actives_smi_map = smi_file_to_dict(smi_a_file)
        inactives_smi_map = smi_file_to_dict(smi_ia_file)
        
        # --- OPTIMIZED: Use pre-computed map ---
        active_ligs = master_glob_map.get(a_in, [])
        inactive_ligs = master_glob_map.get(ia_in, [])
        # ---

        if not actives_smi_map and active_ligs:
            print(f"Warning: No SMILES found in {smi_a_file.name} for {targets[i]}")
        if not inactives_smi_map and inactive_ligs:
            print(f"Warning: No SMILES found in {smi_ia_file.name} for {targets[i]}")

        # Process Actives (now iterates over an in-memory list)
        for lig in active_ligs:
            lig_stem = lig.stem
            template_smi = actives_smi_map.get(lig_stem)
            if not template_smi:
                continue # Skip if no matching SMILES was found

            out = a_out / f"{lig_stem}.sdf"
            local.append((targets[i], receptors[i], lig.resolve(), template_smi, out.resolve()))
            if test_run: break
        
        if test_run: return local # only do one active if test_run

        # Process Inactives (now iterates over an in-memory list)
        for lig in inactive_ligs:
            lig_stem = lig.stem
            template_smi = inactives_smi_map.get(lig_stem)
            if not template_smi:
                continue # Skip if no matching SMILES was found
                
            out = ia_out / f"{lig_stem}.sdf"
            local.append((targets[i], receptors[i], lig.resolve(), template_smi, out.resolve()))
            if test_run: break
            
        return local

    # This tmap is now much faster as process_box does no I/O
    per_box_lists = tmap(
        process_box, 
        list(range(n_boxes)), 
        max_workers=workers, 
        desc="Collecting boxes"
    )
    entries = []
    for lst in per_box_lists:
        entries.extend(lst)
    return entries

def _make_job_from_entry(entry):
    target, receptor, ligpath, template_smi, outpath = entry
    return {
        "target": str(target),
        "receptor": str(receptor),
        "ligand": str(ligpath),
        "ligand_name": Path(ligpath).stem,
        "template_smi": str(template_smi), # <-- UPDATED
        "output_file": str(outpath),
    }

def prepare_master(boxes_csv: Path, master_manifest: Path, shard_total: int, test_run: bool,
                   prepare_workers: int):
    
    workers = prepare_workers or min(32, max(1, multiprocessing.cpu_count()))
    entries = collect_ligand_entries(boxes_csv, test_run=test_run, workers=workers)
    n = len(entries)
    
    if n == 0:
        # Added template_smi to header
        fieldnames = ["task","Done?","target","receptor","ligand","ligand_name","template_smi","output_file"]
        master_manifest.parent.mkdir(parents=True, exist_ok=True)
        tmp = master_manifest.with_suffix(".tmp")
        with open(tmp, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
        tmp.replace(master_manifest)
        print(f"Wrote empty master manifest: {master_manifest}")
        return

    # --- OPTIMIZATION 2: Replaced Path.exists() loop ---

    # 1. Build jobs (Use a list comprehension, it's faster for this simple task)
    print("Building job specifications...")
    jobs = [_make_job_from_entry(e) for e in tqdm(entries, desc="Building jobs")]

    # 2. Get all unique output directories from the jobs
    output_dirs = set(Path(j["output_file"]).parent for j in jobs)

    # 3. Define a helper to glob one directory
    def glob_dir(d: Path):
        if not d.is_dir(): # Handle cases where dir hasn't been created
            return set()
        return set(d.glob("*.sdf"))

    # 4. Glob all output directories IN PARALLEL
    print(f"Scanning {len(output_dirs)} output directories for existing SDFs...")
    existing_sdfs_lists = tmap(glob_dir, list(output_dirs), max_workers=workers, desc="Scanning for existing SDFs")

    # 5. Flatten the results into a single set for fast lookup
    existing_sdfs = set()
    for sdf_set in existing_sdfs_lists:
        existing_sdfs.update(sdf_set)
    print(f"Found {len(existing_sdfs)} existing SDF files.")

    # 6. Check for done flags using a fast, in-memory set lookup
    # (This is now CPU-bound and instant, no tmap needed)
    print("Checking outputs...")
    done_flags = [Path(j["output_file"]) in existing_sdfs for j in tqdm(jobs, desc="Checking outputs")]
    
    # --- END OF OPTIMIZATION ---

    rows = []
    undone_indices = []
    for i, job in enumerate(jobs):
        done = "Yes" if done_flags[i] else "No"
        row = {"task": "", "Done?": done}
        row.update(job)
        rows.append(row)
        if done == "No":
            undone_indices.append(i)

    m = len(undone_indices)
    base = m // shard_total if shard_total > 0 else 0
    rem = m % shard_total if shard_total > 0 else 0
    blocks = []
    start = 0
    for t in range(shard_total):
        size = base + (1 if t < rem else 0)
        blocks.append((start, start+size))
        start += size

    for assign_idx in range(m):
        global_idx = undone_indices[assign_idx]
        task = 0
        for t,(s,e) in enumerate(blocks):
            if s <= assign_idx < e:
                task = t
                break
        rows[global_idx]["task"] = str(task)

    fieldnames = list(rows[0].keys()) # <-- Will now automatically include 'template_smi'
    master_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = master_manifest.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k,"") for k in fieldnames})
    tmp.replace(master_manifest)
    todo_count = sum(1 for r in rows if r["Done?"] == "No")
    print(f"Wrote master manifest: {master_manifest} total={n} todo={todo_count}")

def read_master(master_manifest: Path):
    rows = []
    with open(master_manifest, newline="") as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            rows.append(r)
    return rows
def convert_pdbqt_to_sdf(input_pdbqt: Path, output_sdf: Path, template_smi_string: str):
    """
    RDKit conversion using SMILES template. Attempts to handle H-mismatch:
    - builds template_with_H and template_no_H and picks the one whose atom
      count matches the parsed pose_mol.
    Returns (rc, message).
    """
    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    input_pdbqt = Path(input_pdbqt)
    output_sdf = Path(output_sdf)

    # prepare log
    log_dir = output_sdf.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{input_pdbqt.stem}.log"

    def _log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        full = f"[{ts}] {msg}\n"
        try:
            with open(log_file, "a", encoding="utf-8") as L:
                L.write(full)
        except Exception:
            print(full, end="", flush=True)

    _log(f"convert START in={input_pdbqt} out={output_sdf}")
    _log("TEMPLATE: Using SMILES string from manifest")

    # 1) Load template from SMILES and build H variants
    try:
        smi = template_smi_string
        if not smi:
            _log("ERR template SMI string is empty")
            return (51, "template SMI is empty")

        template_base = Chem.MolFromSmiles(smi)
        if template_base is None:
            _log(f"ERR RDKit failed to parse SMI: {smi[:100]}")
            return (52, "RDKit MolFromSmiles failed")

        # canonicalize/conformer-free template
        # Make an explicit-H version and then also derive a no-H version.
        template_with_H = Chem.AddHs(template_base, addCoords=True)
        template_no_H = Chem.RemoveHs(template_with_H, sanitize=False)

        # ensure sanitization for template variants (safe)
        Chem.SanitizeMol(template_with_H)
        Chem.SanitizeMol(template_no_H)

    except Exception as e:
        _log(f"ERR loading template from SMI: {e}")
        return (53, f"template-load-exc: {e}")

    # 2) Read PDBQT and convert into a PDB-like block and parse pose (no bond orders)
    try:
        pose_lines = []
        found_model = False
        with open(input_pdbqt, 'r') as f_in:
            for line in f_in:
                if line.startswith('MODEL'):
                    found_model = True
                if found_model:
                    pose_lines.append(line)
                if line.startswith('ENDMDL'):
                    break

        if not pose_lines and not found_model:
            _log("No MODEL/ENDMDL found, reading as single pose")
            with open(input_pdbqt, 'r') as f_in:
                pose_lines = f_in.readlines()

        pose_block = "".join(pose_lines)
        if not pose_block:
            _log("ERR PDBQT file appears empty or no ATOM/HETATM lines found")
            return (60, "PDBQT file empty or no ATOM/HETATM lines")

        # Convert PDBQT -> crude PDB block. Keep ATOM/HETATM/TER/ENDMDL and trim trailing columns.
        pdb_lines = []
        for L in pose_block.splitlines():
            if L.startswith(("ATOM  ", "HETATM", "TER   ", "ENDMDL")):
                # keep standard PDB columns up to coords. Truncate any AutoDock extras.
                pdb_lines.append(L[:54].ljust(54))
        pdb_block = "\n".join(pdb_lines) + "\n"

        if not pdb_block.strip():
            _log("ERR no ATOM/HETATM lines after converting PDBQT -> PDB block")
            return (60, "no atom lines after conversion")

        pose_mol = rdmolfiles.MolFromPDBBlock(pdb_block, removeHs=False)
        if pose_mol is None:
            # Try parsing after removing any stray non-standard records
            _log("WARN initial MolFromPDBBlock returned None. Trying relaxed parse (removeHs=True).")
            pose_mol = rdmolfiles.MolFromPDBBlock(pdb_block, removeHs=True)
            if pose_mol is None:
                _log("ERR RDKit MolFromPDBBlock failed to parse pose")
                return (61, "RDKit MolFromPDBBlock failed")

    except Exception as e:
        _log(f"ERR loading pose: {e}")
        return (62, f"pose-load-exc: {e}")

    pose_count = pose_mol.GetNumAtoms()
    _log(f"Pose parsed. atoms={pose_count}")

    # 3) Choose template variant matching pose atom count
    chosen_template = None
    reason = ""
    try:
        wH_count = template_with_H.GetNumAtoms()
        nH_count = template_no_H.GetNumAtoms()
        _log(f"Template atoms: with_H={wH_count} no_H={nH_count}")

        if wH_count == pose_count:
            chosen_template = template_with_H
            reason = "match_with_H"
        elif nH_count == pose_count:
            chosen_template = template_no_H
            reason = "match_no_H"
        else:
            # fallback: try matching heavy-atom counts
            tmpl_heavy = template_no_H.GetNumHeavyAtoms()
            pose_heavy = sum(1 for a in pose_mol.GetAtoms() if a.GetAtomicNum() > 1)
            _log(f"Heavy atoms: template_noH={tmpl_heavy} pose={pose_heavy}")
            if tmpl_heavy == pose_heavy:
                chosen_template = template_no_H
                reason = "match_heavy_no_H"
            else:
                # try adding Hs to pose and re-check (best-effort)
                _log("No atom-count match. Attempting to add implicit Hs to template_no_H and re-evaluate.")
                try:
                    # create an RDKit copy and add Hs then sanitize
                    tmp = Chem.AddHs(Chem.Mol(template_no_H), addCoords=True)
                    Chem.SanitizeMol(tmp)
                    if tmp.GetNumAtoms() == pose_count:
                        chosen_template = tmp
                        reason = "added_H_to_template_on_the_fly"
                    else:
                        # give up with informative error
                        _log("Unable to find template variant matching pose atom count.")
                        _log(f"Counts: pose={pose_count} template_with_H={wH_count} template_no_H={nH_count}")
                        return (70, f"atom-count-mismatch pose={pose_count} tmpl_with_H={wH_count} tmpl_no_H={nH_count}")
                except Exception as e:
                    _log(f"ERR while trying AddHs fallback: {e}")
                    return (71, f"hydrogen-fallback-exc:{e}")
    except Exception as e:
        _log(f"ERR while selecting template variant: {e}")
        return (72, f"template-select-exc:{e}")

    _log(f"Selected template variant: {reason} (template_atoms={chosen_template.GetNumAtoms()})")

    # 4) Assign bond orders from chosen template
    try:
        final_mol = AllChem.AssignBondOrdersFromTemplate(chosen_template, pose_mol)
        if final_mol is None or final_mol.GetNumAtoms() == 0:
            _log("ERR AssignBondOrdersFromTemplate returned empty molecule. Atom ordering or counts may differ.")
            _log(f"Template atoms: {chosen_template.GetNumAtoms()} Pose atoms: {pose_mol.GetNumAtoms()}")
            return (80, "AssignBondOrdersFromTemplate atom mismatch")
        final_mol.SetProp("_Name", input_pdbqt.stem)
    except Exception as e:
        _log(f"ERR AssignBondOrdersFromTemplate failed: {e}")
        try:
            _log(f"Template atoms: {chosen_template.GetNumAtoms()} Pose atoms: {pose_mol.GetNumAtoms()}")
        except Exception:
            pass
        return (81, f"template-assign-exc: {e}")

    # 5) Write SDF
    try:
        with Chem.SDWriter(str(output_sdf)) as w:
            w.write(final_mol)
    except Exception as e:
        _log(f"ERR writing SDF: {e}")
        return (31, f"sdf-write-exc: {e}")

    # 6) Verify file
    try:
        if not output_sdf.exists():
            _log("ERR sdf file not created")
            return (40, "sdf file not created")
        size = output_sdf.stat().st_size
        if size == 0:
            _log("ERR sdf file zero bytes")
            return (41, "sdf file zero bytes")
    except Exception as e:
        _log(f"ERR verifying SDF: {e}")
        return (42, f"sdf-verify-exc: {e}")

    _log(f"SUCCESS wrote {output_sdf} size={size} selected_model=1")
    return (0, f"wrote {output_sdf} size={size} selected_model=1")

def runtime_execute(myrows, runtime_workers):
    def wrapper(job):
        outp = Path(job.get("output_file",""))
        inp = Path(job.get("ligand",""))
        template_smi_string = job.get("template_smi", "") # <-- UPDATED

        if outp.exists():
            return (job, 0, 0.0, "exists")
        
        if not template_smi_string: # <-- UPDATED Check
            return (job, -2, 0.0, f"Template_SMI_missing_in_manifest")
            
        start = time.time()
        # Call updated function
        rc, txt = convert_pdbqt_to_sdf(inp, outp, template_smi_string) 
        return (job, rc, round(time.time()-start,2), txt or "")

    max_workers = runtime_workers or max(1, multiprocessing.cpu_count())
    results_map = {}
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, job in enumerate(myrows):
            fut = ex.submit(wrapper, job)
            futures_map[fut] = i

        with tqdm(total=len(myrows), desc="Converting") as pbar:
            for fut in as_completed(futures_map):
                idx = futures_map[fut]
                job = myrows[idx]
                try:
                    res = fut.result()
                except Exception as e:
                    res = (job, -1, 0.0, f"EXC:{e}")
                results_map[idx] = res
                pbar.set_description(f"Converting {Path(job.get('ligand','')).name}")
                pbar.update(1)

    results = [results_map[i] for i in range(len(myrows))]
    return results

def main():
    args = parse_args()
    boxes = Path(args.csv)
    master = Path(args.master_manifest)

    if args.prepare_master:
        if not boxes.exists():
            raise SystemExit(f"Boxes CSV missing: {boxes}")
        prepare_master(boxes, master, shard_total=args.shard_total, test_run=args.test_run,
                       prepare_workers=args.prepare_workers)
        return

    if not master.exists():
        raise SystemExit("Master manifest missing. Run --prepare_master first.")

    task_id = args.task_id if args.task_id >= 0 else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0") or 0)
    rows = read_master(master)
    myrows = [r for r in rows if r.get("task","") == str(task_id) and r.get("Done?","No") == "No"]
    if not myrows:
        print(f"No todo rows for task {task_id}."); return

    runtime_workers = args.runtime_workers or max(1, multiprocessing.cpu_count())
    # Updated print message
    print(f"[task {task_id}] executing {len(myrows)} jobs with {runtime_workers} worker(s) using RDKit+SMI_Template")
    results = runtime_execute(myrows, runtime_workers)

    ok = sum(1 for (_, rc, _, _) in results if rc == 0)
    err = sum(1 for (_, rc, _, _) in results if rc != 0)
    print(f"Completed. OK={ok} ERR={err}")

if __name__ == "__main__":
    main()