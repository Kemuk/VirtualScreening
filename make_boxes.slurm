#!/bin/bash
#SBATCH --job-name=make_boxes
#SBATCH --output=make_boxes_%j.out
#SBATCH --error=make_boxes_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail
set -x

# Modules
module purge || true
module load PyMOL || module load pymol || true
command -v pymol >/dev/null 2>&1 || { echo "ERROR: PyMOL not found"; exit 1; }
module load OpenBabel || true   # convert pdbqt -> pdb if needed

# --- Box sizing controls ---
PAD="${PAD:-8.0}"          # padding per axis (Å)
MIN_SIZE="${MIN_SIZE:-18}" # minimum edge length (Å)
MAX_SIZE="${MAX_SIZE:-36}" # maximum edge length (Å)
THREAD="${THREAD:-8000}"   # QuickVina2-GPU lanes

# Go to dataset root
cd "$SLURM_SUBMIT_DIR/LIT_PCBA" || { echo "ERROR: LIT_PCBA not found"; exit 1; }

# Master CSV
MASTER_CSV="$SLURM_SUBMIT_DIR/LIT_PCBA/vina_boxes.csv"
echo "target,receptor_pdbqt,actives_glob,inactives_glob,out_dir,center_x,center_y,center_z,size_x,size_y,size_z,thread,rescoring_receptor_pdb,sdf_out_actives_dir,sdf_out_inactives_dir,aev_plig_csv" > "$MASTER_CSV"

# Inline PyMOL script to compute box
PY_SCRIPT="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR}/pymol_box.py"
cat > "$PY_SCRIPT" << 'PYEOF'
from pymol import cmd
import sys
rec, lig, pad, min_size, max_size = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
cmd.reinitialize()
cmd.load(rec, "prot")
cmd.load(lig, "lig")
sel = "lig and not hydro"
cx, cy, cz = cmd.centerofmass(sel)
m = cmd.get_model(sel)
xs = [a.coord[0] for a in m.atom]; ys = [a.coord[1] for a in m.atom]; zs = [a.coord[2] for a in m.atom]
if not xs:
    cx, cy, cz = cmd.centerofmass("prot and not hydro"); sx = sy = sz = 22.0
else:
    clamp = lambda v: max(min_size, min(max_size, v))
    sx = clamp((max(xs)-min(xs)) + 2*pad)
    sy = clamp((max(ys)-min(ys)) + 2*pad)
    sz = clamp((max(zs)-min(zs)) + 2*pad)
print(f"{cx:.3f},{cy:.3f},{cz:.3f},{sx:.3f},{sy:.3f},{sz:.3f}")
PYEOF

# Find all receptors
mapfile -d '' -t RECEPTORS < <(find . -mindepth 2 -maxdepth 2 -type f -name '*_protein.pdbqt' -print0 | sort -z)
echo "Found ${#RECEPTORS[@]} receptors"

for rpq in "${RECEPTORS[@]}"; do
  target="$(basename "$(dirname "$rpq")")"       # e.g. IDH1
  base="$(basename "$rpq")"                      # e.g. 4umx_protein.pdbqt
  complex_name="${base%%_protein.*}"             # e.g. 4umx

  ligand_guess="$(dirname "$rpq")/${complex_name}_ligand.pdbqt"
  if [ ! -s "$ligand_guess" ]; then
    ligand_guess="$(find "$(dirname "$rpq")" -maxdepth 1 -type f -name '*_ligand.pdbqt' | head -n1 || true)"
  fi
  if [ -z "$ligand_guess" ] || [ ! -s "$ligand_guess" ]; then
    echo "WARN: No ligand for $target; skipping."
    continue
  fi

  # Convert receptor & ligand to PDB if possible
  tmpd="$(mktemp -d)"
  rec_pdb="$tmpd/receptor.pdb"
  lig_pdb="$tmpd/ligand.pdb"
  if command -v obabel >/dev/null 2>&1; then
    obabel -ipdbqt "$rpq" -opdb -O "$rec_pdb" >/dev/null 2>&1 || rec_pdb="$rpq"
    obabel -ipdbqt "$ligand_guess" -opdb -O "$lig_pdb" >/dev/null 2>&1 || lig_pdb="$ligand_guess"
  else
    rec_pdb="$rpq"; lig_pdb="$ligand_guess"
  fi

  # Compute box
  IFS=',' read -r cx cy cz sx sy sz < <(pymol -cq "$PY_SCRIPT" -- "$rec_pdb" "$lig_pdb" "$PAD" "$MIN_SIZE" "$MAX_SIZE")
  rm -rf "$tmpd"

  # Globs and outputs
  actives_glob="$(dirname "$rpq")/pdbqt/actives/*.pdbqt"
  inactives_glob="$(dirname "$rpq")/pdbqt/inactives/*.pdbqt"
  out_dir="$(dirname "$rpq")/pdbqt/out"

  rescoring_receptor_pdb="$(dirname "$rpq")/rescoring/receptors_pdb/${target}.pdb"
  sdf_out_actives_dir="$(dirname "$rpq")/rescoring/poses_sdf/actives"
  sdf_out_inactives_dir="$(dirname "$rpq")/rescoring/poses_sdf/inactives"
  aev_plig_csv="$(dirname "$rpq")/rescoring/datasets/aev_plig_${target}.csv"

  mkdir -p "$out_dir" "$(dirname "$rescoring_receptor_pdb")" "$sdf_out_actives_dir" "$sdf_out_inactives_dir" "$(dirname "$aev_plig_csv")"

  # Append to master CSV
  printf "%s,%s,%s,%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%s,%s,%s,%s\n" \
    "$target" "$rpq" "$actives_glob" "$inactives_glob" "$out_dir" \
    "$cx" "$cy" "$cz" "$sx" "$sy" "$sz" "$THREAD" \
    "$rescoring_receptor_pdb" "$sdf_out_actives_dir" "$sdf_out_inactives_dir" "$aev_plig_csv" \
    >> "$MASTER_CSV"

done

echo "Wrote master CSV: $MASTER_CSV"
