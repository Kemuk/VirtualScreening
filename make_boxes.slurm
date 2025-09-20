#!/bin/bash
#SBATCH --job-name=make_boxes
#SBATCH --output=make_boxes_%j.out
#SBATCH --error=make_boxes_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

: <<'PSEUDOCODE'
- Generate docking configurations for each protein.
- Create a master CSV file (`vina_boxes.csv`) with one line per protein, defining paths and box dimensions.
- Build global chunk files (`chunk_all_*.tsv`) containing all config/ligand pairs for a subsequent job array.
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
THREAD="${THREAD:-8000}"   # QVina GPU parallel scale

# --------------------- Main Setup ---------------------
cd "$SLURM_SUBMIT_DIR/LIT_PCBA" || { echo "ERROR: LIT_PCBA not found"; exit 1; }

# Master CSV with simplified, merged columns
MASTER_CSV="$SLURM_SUBMIT_DIR/LIT_PCBA/vina_boxes.csv"
echo "target,receptor_pdbqt,ligands_pdbqt,out_dir,center_x,center_y,center_z,size_x,size_y,size_z,thread,docked_vina,docked_sdf,aev_plig_csv" > "$MASTER_CSV"

# PyMOL script for calculating the bounding box
PY_SCRIPT="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR}/pymol_box.py"
cat > "$PY_SCRIPT" << 'PYEOF'
from pymol import cmd
import sys
rec, lig, pad, min_size, max_size = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
cmd.reinitialize()
cmd.load(rec, "prot")
if lig:
    cmd.load(lig, "lig")
    sel = "lig and not hydro"
    m = cmd.get_model(sel)
    if m.atom:
        cx, cy, cz = cmd.centerofmass(sel)
        xs = [a.coord[0] for a in m.atom]; ys = [a.coord[1] for a in m.atom]; zs = [a.coord[2] for a in m.atom]
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

CONFIGS=()
for rpq in "${RECEPTORS[@]}"; do
  protein_dir="$(dirname "$rpq")"
  target="$(basename "$protein_dir")"
  base="$(basename "$rpq")"
  complex_name="${base%%_protein.*}"

  # Find a representative ligand to size the box
  actives_dir="$protein_dir/pdbqt/actives"
  inactives_dir="$protein_dir/pdbqt/inactives"
  mkdir -p "$actives_dir" "$inactives_dir"
  rep_lig="$(find "$actives_dir" -maxdepth 1 -type f -name '*.pdbqt' | head -n1 || true)"
  [ -z "$rep_lig" ] && rep_lig="$(find "$inactives_dir" -maxdepth 1 -type f -name '*.pdbqt' | head -n1 || true)"
  [ -z "$rep_lig" ] && rep_lig="$(find "$protein_dir" -maxdepth 1 -type f -name '*_ligand.pdbqt' | head -n1 || true)"

  # Convert to PDB for PyMOL if possible
  tmpd="$(mktemp -d)"
  rec_pdb="$tmpd/receptor.pdb"
  lig_pdb="$tmpd/ligand.pdb"
  if command -v obabel >/dev/null 2>&1; then
    obabel -ipdbqt "$rpq" -opdb -O "$rec_pdb" >/dev/null 2>&1 || rec_pdb="$rpq"
    if [ -n "$rep_lig" ]; then
      obabel -ipdbqt "$rep_lig" -opdb -O "$lig_pdb" >/dev/null 2>&1 || lig_pdb="$rep_lig"
    else
      lig_pdb=""
    fi
  else
    rec_pdb="$rpq"; lig_pdb="${rep_lig:-}"
  fi

  # Robustly capture PyMOL output
  pymol_out=$(pymol -cq "$PY_SCRIPT" -- "$rec_pdb" "$lig_pdb" "$PAD" "$MIN_SIZE" "$MAX_SIZE")
  rm -rf "$tmpd"
  if [[ -z "$pymol_out" ]]; then
      echo "ERROR: PyMOL returned no output for receptor '$rpq'. Aborting." >&2
      exit 1
  fi
  IFS=',' read -r cx cy cz sx sy sz <<< "$pymol_out"

  # Define paths and globs for the master CSV
  out_dir="$protein_dir/pdbqt/out"
  aev_plig_csv="$protein_dir/rescoring/datasets/aev_plig_${target}.csv"
  ligands_pdbqt="${protein_dir}/pdbqt/{actives,inactives}/*.pdbqt"
  docked_vina="${protein_dir}/docked_vina/{actives,inactives}"
  docked_sdf="${protein_dir}/docked_sdf/{actives,inactives}"

  mkdir -p "$out_dir" "$protein_dir/docked_vina/"{actives,inactives} "$protein_dir/docked_sdf/"{actives,inactives} "$(dirname "$aev_plig_csv")"

  # Write row to vina_boxes.csv (no quoted strings)
  printf "%s,%s,%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%s,%s,%s\n" \
    "$target" "$rpq" "$ligands_pdbqt" "$out_dir" \
    "$cx" "$cy" "$cz" "$sx" "$sy" "$sz" "$THREAD" \
    "$docked_vina" "$docked_sdf" "$aev_plig_csv" \
    >> "$MASTER_CSV"

  # Generate config files for actives and inactives
  for mode in actives inactives; do
    cfg_path="$protein_dir/${complex_name}_config_${mode}.txt"
    receptor_rel="./$(basename "$rpq")"
    lig_dir_rel="./pdbqt/${mode}/"
    out_dir_rel="./docked_vina/${mode}/"
    LC_ALL=C printf "receptor = %s\nligand_directory =  %s\nout = %s\ncenter_x = %.3f\ncenter_y = %.3f\ncenter_z = %.3f\nsize_x = %.3f\nsize_y = %.3f\nsize_z = %.3f\nthread = %s\n" \
      "$receptor_rel" "$lig_dir_rel" "$out_dir_rel" \
      "$cx" "$cy" "$cz" "$sx" "$sy" "$sz" "$THREAD" > "$cfg_path"
    echo "Wrote config: $cfg_path"
    CONFIGS+=("$cfg_path")
  done
done

echo "Wrote master CSV: $MASTER_CSV"

# ----------------- Build Global Chunk Files -----------------
CHUNKS_DIR="$SLURM_SUBMIT_DIR/LIT_PCBA/chunks"
mkdir -p "$CHUNKS_DIR"
MAX_ARRAY_SIZE="${MAX_ARRAY_SIZE:-1001}"

generate_pairs() {
    for CFG in "${CONFIGS[@]}"; do
        lig_rel=$(awk -F'=' '/^ligand_directory/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' "$CFG")
        [ -z "$lig_rel" ] && { echo "WARN: no ligand_directory in $CFG" >&2; continue; }

        cfg_dir=$(cd "$(dirname "$CFG")" && pwd -P)
        lig_dir="$cfg_dir/${lig_rel#./}"
        [ -d "$lig_dir" ] || { echo "WARN: lig dir missing $lig_dir for $CFG" >&2; continue; }

        find "$lig_dir" -maxdepth 1 -type f -name '*.pdbqt' -print0 | sort -z | while IFS= read -r -d '' LIG; do
            printf "%s\t%s\n" "$CFG" "$LIG"
        done
    done
}

echo "Calculating total pairs..."
total_pairs=$(generate_pairs | wc -l)
echo "Total pairs: $total_pairs"

if [ "$total_pairs" -eq 0 ]; then
    echo "Nothing to chunk. Exiting."
    exit 0
fi

desired_chunks=$(( total_pairs < MAX_ARRAY_SIZE ? total_pairs : MAX_ARRAY_SIZE ))
echo "MaxArraySize: $MAX_ARRAY_SIZE -> desired_chunks: $desired_chunks"

echo "Generating chunk files in $CHUNKS_DIR..."
generate_pairs | split -n "r/${desired_chunks}" -d --suffix-length=4 --additional-suffix=.tsv - "${CHUNKS_DIR}/chunk_all_"

# --------------------- Report & Summary ---------------------
total_chunks=$(find "$CHUNKS_DIR" -type f -name 'chunk_all_*.tsv' | wc -l)
echo "Chunks dir: $CHUNKS_DIR"
echo "Total chunks created: $total_chunks"
echo "Submit the next step with:"
echo "  sbatch --array=0-$((total_chunks-1)) <your_next_script.slurm>"
