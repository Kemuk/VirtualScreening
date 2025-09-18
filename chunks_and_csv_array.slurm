#!/usr/bin/env bash
# FILE: chunks_pk_minimal.sh
#SBATCH --job-name=chunks_pk_min
#SBATCH --output=logs/chunks_pk_%A_%a.out
#SBATCH --error=logs/chunks_pk_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --array=0-1000

set -euo pipefail
export LC_ALL=C

module purge || true
module load parallel || true

cd "${SLURM_SUBMIT_DIR}/LIT_PCBA" || { echo "ERROR: LIT_PCBA not found"; exit 1; }

CHUNKS_DIR="${CHUNKS_DIR:-chunks}"
mkdir -p "$CHUNKS_DIR" ".sentinels/chunks_pk_min" "logs"

create_chunks_once() {
  # why: ensure idempotent chunking across array tasks
  local lock="$CHUNKS_DIR/.chunks.lock"
  {
    flock -n 9 || return 0
    shopt -s nullglob
    local existing=( "$CHUNKS_DIR"/chunk_*.txt )
    (( ${#existing[@]} > 0 )) && { echo "Chunks exist."; return 0; }

    mapfile -t ROWS < <(
      for pdir in */; do
        [[ -d "$pdir" ]] || continue
        p="${pdir%/}"
        cfg="$p/.placeholder.cfg"; [[ -f "$p"/*.cfg ]] && cfg=$(ls -1 "$p"/*.cfg | head -n1)
        for mode in actives inactives; do
          find "$p" -type f -path "*/$mode/*.pdbqt" -print
          find "$p" -type f \( -path "*/ligands/$mode/*.pdbqt" -o -path "*/input/$mode/*.pdbqt" \) -print
        done | awk -v cfg="$cfg" -v root="$PWD" '{ printf "%s\t%s\n", substr(cfg, length(root)+2), substr($0, length(root)+2) }'
      done
    )
    (( ${#ROWS[@]} == 0 )) && { echo "No inputs."; return 0; }

    local SHARD_SIZE="${SHARD_SIZE:-50000}"
    local total="${#ROWS[@]}"
    local n=$(( (total + SHARD_SIZE - 1) / SHARD_SIZE ))
    local width=${#n}
    local idx=0
    for (( c=0; c<n; c++ )); do
      out="$CHUNKS_DIR/chunk_$(printf "%0${width}d" "$c").txt"
      : > "$out"
      for (( k=0; k<SHARD_SIZE && idx<total; k++,idx++ )); do
        printf "%s\n" "${ROWS[$idx]}" >> "$out"
      done
    done
    echo "Wrote $n chunk files."
  } 9>"$lock"
}
create_chunks_once

# Select this task's chunk
mapfile -t CHUNKS < <(ls -1 "$CHUNKS_DIR"/chunk_*.txt 2>/dev/null || true)
NCHUNKS=${#CHUNKS[@]}
(( NCHUNKS == 0 )) && { echo "No chunks."; exit 0; }
(( SLURM_ARRAY_TASK_ID >= NCHUNKS )) && { touch ".sentinels/chunks_pk_min/task_${SLURM_ARRAY_TASK_ID}.done"; exit 0; }

CHUNK_FILE="${CHUNKS[$SLURM_ARRAY_TASK_ID]}"
OUT_CSV="${CHUNK_FILE%.txt}.csv"

# Header: no rmsd; Best_Score -> pK
echo "Ligand_ID,Protein_ID,Ligand_SDF_Path,Protein_SDF_Path,pK,MW,LogP,HBD,HBA,is_active,pred_active" > "$OUT_CSV"

# --- helpers (simple, no fallbacks, no defaults) ---

extract_pk() {
  # first numeric row with mode == 1, print affinity (col 2)
  awk '/^[[:space:]]*1[[:space:]]/ { print $2; exit }' "$1"
}

process() {
  local cfg="$1" input_pdbqt="$2"
  [[ -f "$input_pdbqt" ]] || return

  local protein_dir; protein_dir=$(dirname "$cfg")
  local protein_id; protein_id=$(basename "$protein_dir")
  local lig_id; lig_id=$(basename "$input_pdbqt" .pdbqt)

  local mode is_active
  if [[ "$input_pdbqt" == *"/actives/"* ]]; then mode="actives"; is_active=1; else mode="inactives"; is_active=0; fi

  local logf="$protein_dir/docked_vina/$mode/log/${lig_id}.log"
  [[ -f "$logf" ]] || return

  local pk; pk=$(extract_pk "$logf")
  [[ -n "$pk" ]] || return  # skip if no mode=1 line

  local lig_sdf="./${protein_id}/docked_sdf/${mode}/${lig_id}.sdf"
  local prot_sdf="./${protein_id}/${protein_id}.sdf"

  # MW/LogP/HBD/HBA/pred_active left empty by design
  printf "%s,%s,%s,%s,%s,,,,,%s,\n" \
    "$lig_id" "$protein_id" "$lig_sdf" "$prot_sdf" "$pk" "$is_active"
}
export -f process extract_pk

# Run this chunk
parallel -j "$SLURM_CPUS_PER_TASK" --colsep '\t' process {1} {2} < "$CHUNK_FILE" >> "$OUT_CSV"
echo "Wrote: $OUT_CSV"

# Combine all on last index
if [[ "${SLURM_ARRAY_TASK_ID}" -eq $((NCHUNKS-1)) ]]; then
  mapfile -t PARTS < <(ls -1 "$CHUNKS_DIR"/chunk_*.csv 2>/dev/null)
  if (( ${#PARTS[@]} > 0 )); then
    head -n1 "${PARTS[0]}" > aev_plig.csv
    tail -n+2 -q "${PARTS[@]}" >> aev_plig.csv
    gzip -f aev_plig.csv
    echo "Final: aev_plig.csv.gz"
  fi
fi

# Mark done
touch ".sentinels/chunks_pk_min/task_${SLURM_ARRAY_TASK_ID}.done"
