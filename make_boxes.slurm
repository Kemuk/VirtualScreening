#!/bin/bash
#SBATCH --job-name=make_boxes
#SBATCH --output=make_boxes_%j.out
#SBATCH --error=make_boxes_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

: <<'PSEUDOCODE'
- Generate docking configurations for each protein.
- Create a master CSV file (`vina_boxes.csv`) with one line per protein.
- The CSV will have separate, explicit columns for 'actives_dir' and 'inactives_dir'.
PSEUDOCODE

set -euo pipefail
set -x

# --------------------- Modules & Environment ---------------------
module purge || true
module load PyMOL || module load pymol || true
command -v pymol >/dev/null 2>&1 || { echo "ERROR: PyMOL not found"; exit 1; }
module load OpenBabel || true

# --------------------- Box Sizing Controls ---------------------
PAD="${PAD:-8.0}"
MIN_SIZE="${MIN_SIZE:-18}"
MAX_SIZE="${MAX_SIZE:-36}"
THREAD="${THREAD:-8000}"

# --------------------- Main Setup ---------------------
cd "$SLURM_SUBMIT_DIR/LIT_PCBA" || { echo "ERROR: LIT_PCBA not found"; exit 1; }

# --- MODIFIED: Master CSV with explicit actives_dir and inactives_dir columns ---
MASTER_CSV="$SLURM_SUBMIT_DIR/LIT_PCBA/vina_boxes.csv"
echo "target,receptor_pdbqt,actives_dir,inactives_dir,out_dir,center_x,center_y,center_z,size_x,size_y,size_z,thread,docked_vina,docked_sdf,aev_plig_csv" > "$MASTER_CSV"

# PyMOL script for calculating the bounding box
PY_SCRIPT="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR}/pymol_box.py"
cat > "$PY_SCRIPT" << 'PYEOF'
from pymol import cmd, stored
import sys
rec, lig, pad, min_size, max_size = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
cmd.reinitialize()
cmd.load(rec, "prot")
if lig:
    cmd.load(lig, "lig")
    sel = "lig and not hydro"
    stored.m = []
    cmd.iterate(sel, "stored.m.append(1)")
    if stored.m:
        cx, cy, cz = cmd.centerofmass(sel)
        xs, ys, zs = [], [], []
        for a in cmd.get_model(sel).atom:
            xs.append(a.coord[0]); ys.append(a.coord[1]); zs.append(a.coord[2])
        clamp = lambda v: max(min_size, min(max_size, v))
        sx = clamp((max(xs)-min(xs)) + 2*pad)
        sy = clamp((max(ys)-min(ys)) + 2*pad)
        sz = clamp((max(zs)-min(zs)) + 2*pad)
        print(f"{cx:.3f},{cy:.3f},{cz:.3f},{sx:.3f},{sy:.3f},{sz:.3f}")
        sys.exit(0)
cx, cy, cz = cmd.centerofmass("prot and not hydro")
print(f"{cx:.3f},{cy:.3f},{cz:.3f},22.000,22.000,22.000")
PYEOF

# --------------------- Receptor Processing Loop ---------------------
mapfile -d '' -t RECEPTORS < <(find . -mindepth 2 -maxdepth 2 -type f -name '*_protein.pdbqt' -print0 | sort -z)
echo "Found ${#RECEPTORS[@]} receptors"

for rpq in "${RECEPTORS[@]}"; do
    protein_dir="$(dirname "$rpq")"
    target="$(basename "$protein_dir")"

    # Define explicit paths for actives and inactives directories
    actives_dir="$protein_dir/pdbqt/actives"
    inactives_dir="$protein_dir/pdbqt/inactives"
    mkdir -p "$actives_dir" "$inactives_dir"

    # Find a representative ligand to size the box
    rep_lig="$(find "$actives_dir" "$inactives_dir" -maxdepth 1 -type f -name '*.pdbqt' 2>/dev/null | head -n1 || true)"
    [ -z "$rep_lig" ] && rep_lig="$(find "$protein_dir" -maxdepth 1 -type f -name '*_ligand.pdbqt' | head -n1 || true)"

    # Get Box Dimensions using PyMOL
    tmpd="$(mktemp -d)"
    rec_pdb="$tmpd/receptor.pdb"
    lig_pdb="$tmpd/ligand.pdb"
    obabel -ipdbqt "$rpq" -opdb -O "$rec_pdb" &>/dev/null || rec_pdb="$rpq"
    if [ -n "$rep_lig" ]; then
        obabel -ipdbqt "$rep_lig" -opdb -O "$lig_pdb" &>/dev/null || lig_pdb="$rep_lig"
    else
        lig_pdb=""
    fi
    pymol_out=$(pymol -cq "$PY_SCRIPT" -- "$rec_pdb" "$lig_pdb" "$PAD" "$MIN_SIZE" "$MAX_SIZE")
    rm -rf "$tmpd"
    IFS=',' read -r cx cy cz sx sy sz <<< "$pymol_out"

    # Define other paths for the master CSV
    out_dir="$protein_dir/pdbqt/out"
    aev_plig_csv="$protein_dir/rescoring/datasets/aev_plig_${target}.csv"
    # These paths still use brace expansion for mkdir, which is fine for the shell
    docked_vina="${protein_dir}/docked_vina/{actives,inactives}"
    docked_sdf="${protein_dir}/docked_sdf/{actives,inactives}"

    # Ensure all directories exist
    mkdir -p "$out_dir" "$protein_dir/docked_vina/"{actives,inactives} "$protein_dir/docked_sdf/"{actives,inactives} "$(dirname "$aev_plig_csv")"

    # --- MODIFIED: Write one row per protein with explicit directory columns ---
    printf "%s,%s,%s,%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%s,%s,%s\n" \
        "$target" "$rpq" "$actives_dir" "$inactives_dir" "$out_dir" \
        "$cx" "$cy" "$cz" "$sx" "$sy" "$sz" "$THREAD" \
        "${docked_vina//\{*\}/}" "${docked_sdf//\{*\}/}" "$aev_plig_csv" \
        >> "$MASTER_CSV"

done

echo "Wrote master CSV: $MASTER_CSV"