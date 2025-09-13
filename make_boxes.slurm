#!/bin/bash
#SBATCH --job-name=make_boxes
#SBATCH --output=make_boxes_%j.out
#SBATCH --error=make_boxes_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

: <<'PSEUDOCODE'
- Generate two configs per protein: <complex>_config_actives.txt / _inactives.txt
  receptor=./<receptor>.pdbqt, ligand_directory=./pdbqt/<mode>/, out=./docked_vina/<mode>/
  include center/size/thread
- Build per-config chunks in LIT_PCBA/chunks:
  Each chunk file contains lines: CONFIG<TAB>LIGAND
  Chunk filenames: chunk_<target>_<complex>_<mode>_<NNNN>.txt (0-based index)
- Emit summary + sbatch array hint
PSEUDOCODE

set -euo pipefail
set -x

# Modules
module purge || true
module load PyMOL || module load pymol || true
command -v pymol >/dev/null 2>&1 || { echo "ERROR: PyMOL not found"; exit 1; }
module load OpenBabel || true

# Box sizing controls
PAD="${PAD:-8.0}"
MIN_SIZE="${MIN_SIZE:-18}"
MAX_SIZE="${MAX_SIZE:-36}"
THREAD="${THREAD:-8000}"   # QVina GPU parallel scale

# Go to dataset root
cd "$SLURM_SUBMIT_DIR/LIT_PCBA" || { echo "ERROR: LIT_PCBA not found"; exit 1; }

# Master CSV
MASTER_CSV="$SLURM_SUBMIT_DIR/LIT_PCBA/vina_boxes.csv"
echo "target,receptor_pdbqt,actives_glob,inactives_glob,out_dir,center_x,center_y,center_z,size_x,size_y,size_z,thread,rescoring_receptor_pdb,sdf_out_actives_dir,sdf_out_inactives_dir,aev_plig_csv" > "$MASTER_CSV"

# PyMOL script
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

# Find receptors (two levels down)
mapfile -d '' -t RECEPTORS < <(find . -mindepth 2 -maxdepth 2 -type f -name '*_protein.pdbqt' -print0 | sort -z)
echo "Found ${#RECEPTORS[@]} receptors"

CONFIGS=()

for rpq in "${RECEPTORS[@]}"; do
  protein_dir="$(dirname "$rpq")"           # ./IDH1
  target="$(basename "$protein_dir")"       # IDH1
  base="$(basename "$rpq")"                 # 4umx_protein.pdbqt
  complex_name="${base%%_protein.*}"        # 4umx

  actives_dir="$protein_dir/pdbqt/actives"
  inactives_dir="$protein_dir/pdbqt/inactives"
  mkdir -p "$actives_dir" "$inactives_dir"

  # find representative ligand for sizing
  rep_lig="$(find "$actives_dir" -maxdepth 1 -type f -name '*.pdbqt' | head -n1 || true)"
  [ -z "$rep_lig" ] && rep_lig="$(find "$inactives_dir" -maxdepth 1 -type f -name '*.pdbqt' | head -n1 || true)"
  [ -z "$rep_lig" ] && rep_lig="$(find "$protein_dir" -maxdepth 1 -type f -name '*_ligand.pdbqt' | head -n1 || true)"

  # convert to PDB if possible for PyMOL
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

  IFS=',' read -r cx cy cz sx sy sz < <(pymol -cq "$PY_SCRIPT" -- "$rec_pdb" "$lig_pdb" "$PAD" "$MIN_SIZE" "$MAX_SIZE")
  rm -rf "$tmpd"

  # CSV bookkeeping
  actives_glob="$actives_dir/*.pdbqt"
  inactives_glob="$inactives_dir/*.pdbqt"
  out_dir="$protein_dir/pdbqt/out"

  rescoring_receptor_pdb="$protein_dir/rescoring/receptors_pdb/${target}.pdb"
  sdf_out_actives_dir="$protein_dir/rescoring/poses_sdf/actives"
  sdf_out_inactives_dir="$protein_dir/rescoring/poses_sdf/inactives"
  aev_plig_csv="$protein_dir/rescoring/datasets/aev_plig_${target}.csv"

  mkdir -p "$out_dir" "$(dirname "$rescoring_receptor_pdb")" "$sdf_out_actives_dir" "$sdf_out_inactives_dir" "$(dirname "$aev_plig_csv")" "$protein_dir/docked_vina/actives" "$protein_dir/docked_vina/inactives"

  printf "%s,%s,%s,%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%s,%s,%s,%s\n" \
    "$target" "$rpq" "$actives_glob" "$inactives_glob" "$out_dir" \
    "$cx" "$cy" "$cz" "$sx" "$sy" "$sz" "$THREAD" \
    "$rescoring_receptor_pdb" "$sdf_out_actives_dir" "$sdf_out_inactives_dir" "$aev_plig_csv" \
    >> "$MASTER_CSV"

  # two configs
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

# ----------------- Build PER-CONFIG chunks -----------------
CHUNK_SIZE="${CHUNK_SIZE:-2000}"           # 1000–2000 typical
CHUNKS_DIR="$SLURM_SUBMIT_DIR/LIT_PCBA/chunks"
mkdir -p "$CHUNKS_DIR"

total_pairs=0
total_chunks=0

for CFG in "${CONFIGS[@]}"; do
  cfg_dir="$(cd "$(dirname "$CFG")" && pwd -P)"
  cfg_base="$(basename "$CFG")"                      # e.g., 4umx_config_actives.txt
  complex="${cfg_base%%_config_*}"                   # 4umx
  mode="${cfg_base##*_config_}"; mode="${mode%.*}"   # actives|inactives
  target="$(basename "$cfg_dir")"                    # protein dir name

  lig_rel="$(awk -F'=' '$1 ~ /^ligand_directory[[:space:]]*$/ { $1=""; sub(/^=/,""); print }' "$CFG" | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')"
  [ -z "$lig_rel" ] && { echo "WARN: no ligand_directory in $CFG"; continue; }
  lig_dir="$cfg_dir/${lig_rel#./}"; lig_dir="${lig_dir%/}"
  [ -d "$lig_dir" ] || { echo "WARN: lig dir missing $lig_dir for $CFG"; continue; }

  # enumerate ligands (sorted)
  mapfile -d '' -t ligs < <(find "$lig_dir" -maxdepth 1 -type f -name '*.pdbqt' -print0 | sort -z)
  count="${#ligs[@]}"
  [ "$count" -gt 0 ] || { echo "WARN: 0 ligands for $CFG"; continue; }

  # write chunks
  n_chunks=$(( (count + CHUNK_SIZE - 1) / CHUNK_SIZE ))
  for ((i=0;i<n_chunks;i++)); do
    chunk_path="$CHUNKS_DIR/chunk_${target}_${complex}_${mode}_$(printf "%04d" "$i").txt"
    : > "$chunk_path"
    start=$(( i * CHUNK_SIZE ))
    end=$(( start + CHUNK_SIZE ))
    (( end > count )) && end=$count
    for ((j=start;j<end;j++)); do
      printf "%s\t%s\n" "$CFG" "${ligs[j]}" >> "$chunk_path"
    done
    echo "Wrote $chunk_path ($(wc -l < "$chunk_path") lines)"
  done

  total_pairs=$(( total_pairs + count ))
  total_chunks=$(( total_chunks + n_chunks ))
done

echo "Chunks dir: $CHUNKS_DIR"
echo "Total pairs: $total_pairs"
echo "Total chunks: $total_chunks"
echo "Submit with:"
echo "  N=\$(ls -1 \"$CHUNKS_DIR\"/chunk_*.txt | wc -l); sbatch --array=0-\$((N-1)) qvina_chunks_array.slurm \"$CHUNKS_DIR\""
