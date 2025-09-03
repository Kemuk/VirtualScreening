#!/bin/bash
#SBATCH --job-name=make_boxes
#SBATCH --output=make_boxes_%j.out
#SBATCH --error=make_boxes_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail
set -x

# Modules
module purge || true
module load PyMOL || module load pymol || true
command -v pymol >/dev/null 2>&1 || { echo "ERROR: PyMOL not found"; exit 1; }
module load OpenBabel || true   # used to convert PDBQT->PDB for robust PyMOL parsing

# --- Adaptive sizing controls ---
PAD="${PAD:-8.0}"          # Å padding added on EACH axis
MIN_SIZE="${MIN_SIZE:-18}" # clamp each axis to at least this (Å)
MAX_SIZE="${MAX_SIZE:-36}" # clamp each axis to at most this (Å)

# Go to dataset root: submit from parent of LIT_PCBA/
cd "$SLURM_SUBMIT_DIR/LIT_PCBA" || { echo "ERROR: LIT_PCBA not found in submit dir"; exit 1; }

# Master CSV
MASTER_CSV="$SLURM_SUBMIT_DIR/LIT_PCBA/vina_boxes.csv"
echo "protein_name,complex_name,center_x,center_y,center_z,size_x,size_y,size_z" > "$MASTER_CSV"

# Inline PyMOL helper: prints "cx,cy,cz,sx,sy,sz"
PY_SCRIPT="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR}/pymol_box_from_lig.py"
cat > "$PY_SCRIPT" << 'PYEOF'
from pymol import cmd
import sys, os, math

# Args:
# 1: receptor.pdb (or .pdbqt)
# 2: ligand.pdb (or .pdbqt)  -- REQUIRED for adaptive sizing & center
# 3: pad
# 4: min_size
# 5: max_size

rec = sys.argv[1]
lig = sys.argv[2]
pad = float(sys.argv[3])
min_size = float(sys.argv[4])
max_size = float(sys.argv[5])

cmd.reinitialize()
cmd.load(rec, "prot")
cmd.load(lig, "lig")

sel = "lig and not hydro"
# center = COM of reference ligand (heavy atoms)
cx, cy, cz = cmd.centerofmass(sel)

m = cmd.get_model(sel)
xs = [a.coord[0] for a in m.atom]
ys = [a.coord[1] for a in m.atom]
zs = [a.coord[2] for a in m.atom]
if not xs:
    # if ligand somehow empty, fall back to protein COM and fixed 22 Å cube
    cx, cy, cz = cmd.centerofmass("prot and not hydro")
    sx = sy = sz = 22.0
else:
    def clamp(v): return max(min_size, min(max_size, v))
    spanx = (max(xs) - min(xs)) + 2.0*pad
    spany = (max(ys) - min(ys)) + 2.0*pad
    spanz = (max(zs) - min(zs)) + 2.0*pad
    sx, sy, sz = clamp(spanx), clamp(spany), clamp(spanz)

print(f"{cx:.3f},{cy:.3f},{cz:.3f},{sx:.3f},{sy:.3f},{sz:.3f}")
PYEOF

# Find all receptors once
mapfile -d '' -t RECEPTORS < <(find . -mindepth 2 -maxdepth 2 -type f -name '*_protein.pdbqt' -print0 | sort -z)
echo "Found ${#RECEPTORS[@]} *_protein.pdbqt files under $(pwd)"
(( ${#RECEPTORS[@]} > 0 )) || { echo "No receptors found."; exit 2; }

for rpq in "${RECEPTORS[@]}"; do
  protein_dir="$(dirname "$rpq")"
  base="$(basename "$rpq")"
  protein_name="$(basename "$protein_dir")"
  complex_name="${base%%_protein.*}"

  # reference ligand path(s)
  ligand_guess="$protein_dir/${complex_name}_ligand.pdbqt"
  if [ -s "$ligand_guess" ]; then
    lpq="$ligand_guess"
  else
    # fallback: first *_ligand.pdbqt in the same folder
    lpq="$(find "$protein_dir" -maxdepth 1 -type f -name '*_ligand.pdbqt' | head -n1 || true)"
  fi
  if [ -z "${lpq:-}" ] || [ ! -s "$lpq" ]; then
    echo "WARN: No *_ligand.pdbqt for $complex_name in $protein_dir; skipping."
    continue
  fi

  # Convert receptor & ligand to PDB for PyMOL if possible
  tmpd="$(mktemp -d)"
  rec_pdb="$tmpd/receptor.pdb"
  lig_pdb="$tmpd/ligand.pdb"
  if command -v obabel >/dev/null 2>&1; then
    obabel -ipdbqt "$rpq" -opdb -O "$rec_pdb" >/dev/null 2>&1 || rec_pdb="$rpq"
    obabel -ipdbqt "$lpq" -opdb -O "$lig_pdb" >/dev/null 2>&1 || lig_pdb="$lpq"
  else
    rec_pdb="$rpq"; lig_pdb="$lpq"
  fi

  # Compute center & size from ligand
  IFS=',' read -r cx cy cz sx sy sz < <(pymol -cq "$PY_SCRIPT" -- "$rec_pdb" "$lig_pdb" "$PAD" "$MIN_SIZE" "$MAX_SIZE")

  rm -rf "$tmpd"

  # Per-protein CSV (overwrite each run)
  PROT_CSV="$protein_dir/vina_box.csv"
  {
    echo "protein_name,complex_name,center_x,center_y,center_z,size_x,size_y,size_z"
    printf "%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n" \
      "$protein_name" "$complex_name" "$cx" "$cy" "$cz" "$sx" "$sy" "$sz"
  } > "$PROT_CSV"
  echo "Wrote per-protein CSV: $PROT_CSV"

  # Append to master CSV
  printf "%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n" \
    "$protein_name" "$complex_name" "$cx" "$cy" "$cz" "$sx" "$sy" "$sz" >> "$MASTER_CSV"
done

echo "Wrote master CSV: $MASTER_CSV"
