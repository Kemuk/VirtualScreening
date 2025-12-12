#!/usr/bin/env bash

#SBATCH --job-name=vina_gpu
#SBATCH --output=logs/vina_gpu_%A_%a.out
#SBATCH --error=logs/vina_gpu_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --clusters=htc
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --requeue
#SBATCH --array=1-1000
#SBATCH --mail-user=reub0582@ox.ac.uk   # <<< ADDED: Your email address
#SBATCH --mail-type=BEGIN,END    # <<< ADDED: Notification types

set -u -o pipefail
ulimit -s 8192 || true

module purge || true
module load Boost/1.77.0-GCC-11.2.0 CUDA/12.0.0 || true

cd "${SLURM_SUBMIT_DIR:?}"

QVINA_BIN="${QVINA_BIN:-"./vina-gpu-dev/QuickVina2-GPU-2-1"}"
[[ -x "$QVINA_BIN" ]] || { echo "ERROR: Vina-GPU not found at $(pwd -P)/$QVINA_BIN"; exit 1; }
BINDIR="$(cd "$(dirname "$QVINA_BIN")" && pwd -P)"

DATA_DIR="./LIT_PCBA"
CHUNKS_DIR="$DATA_DIR/chunks"

CHUNK_FILE=$(find "$CHUNKS_DIR" -maxdepth 1 -type f -name 'chunk_all_*.tsv' | sort | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
[[ -f "$CHUNK_FILE" ]] || { echo "ERROR: Chunk file not found for task $SLURM_ARRAY_TASK_ID"; exit 1; }

echo "Processing chunk: $CHUNK_FILE"
echo "SLURM Job ID: ${SLURM_JOB_ID:-na}, Task ID: ${SLURM_ARRAY_TASK_ID:-na}"
echo "Using binary: $QVINA_BIN"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# === Helpers ===
trim(){ sed -E 's/^[[:space:]]+|[[:space:]]+$//g'; }
kv(){
  local key="$1" file="$2"
  awk -v k="^"$key"[[:space:]]*$" -F'=' '
    BEGIN{found=0}
    $1 ~ k {v=$2; gsub(/^[[:space:]]+|[[:space:]]+$/, "", v); print v; found=1}
    END{exit(found?0:1)}
  ' "$file"
}
strip_cr(){ sed 's/\r$//'; }

# === Process Function ===
process_ligand() {
  local CFG_REL="$1"
  local LIG="$2"
  local GPU_ID="$3"

  CFG="$DATA_DIR/${CFG_REL#./}"
  [[ "$LIG" != /* ]] && LIG="$LIG"
  [[ -f "$CFG" && -f "$LIG" ]] || {
    echo "SKIP: Missing CFG or LIG: $CFG, $LIG"
    return
  }

  CFG_DIR="$(cd "$(dirname "$CFG")" && pwd -P)"
  RECEPTOR_REL=$(kv 'receptor' "$CFG" | trim || true)
  OUT_REL=$(kv 'out' "$CFG" | trim || true)
  CX=$(kv 'center_x' "$CFG" | trim || true)
  CY=$(kv 'center_y' "$CFG" | trim || true)
  CZ=$(kv 'center_z' "$CFG" | trim || true)
  SX=$(kv 'size_x' "$CFG" | trim || true)
  SY=$(kv 'size_y' "$CFG" | trim || true)
  SZ=$(kv 'size_z' "$CFG" | trim || true)
  THREAD_VAL=$(kv 'thread' "$CFG" | trim 2>/dev/null || echo "8")

  RECEPTOR="$CFG_DIR/${RECEPTOR_REL#./}"
  OUT_DIR="$CFG_DIR/${OUT_REL#./}"
  LOG_DIR="$OUT_DIR/log"

  mkdir -p "$OUT_DIR" "$LOG_DIR"
  [[ -s "$RECEPTOR" ]] || {
    echo "FATAL: Receptor not found: $RECEPTOR"
    return
  }

  name="$(basename "$LIG")"
  out_pdbqt="$OUT_DIR/$name"
  log="$LOG_DIR/${name%.pdbqt}.log"

  (
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    cd "$BINDIR" && "./$(basename "$QVINA_BIN")" \
      --receptor "$RECEPTOR" \
      --ligand     "$LIG" \
      --center_x "$CX" --center_y "$CY" --center_z "$CZ" \
      --size_x   "$SX" --size_y   "$SY" --size_z   "$SZ" \
      --out        "$out_pdbqt" \
      --seed       42 \
      --thread     "$THREAD_VAL" \
      >"$log" 2>&1

    rc=$?
    if [[ $rc -ne 0 || ! -s "$out_pdbqt" ]]; then
      echo "[FAIL] $(basename "$CFG") :: $name (rc=$rc) -> $log"
    else
      echo "[OK]   $(basename "$CFG") :: $name"
    fi
  ) &
}

# === Main Parallel Loop ===
ok=0; fail=0
gpu_toggle=0

while IFS=$'\t' read -r CFG_REL LIG || [[ -n "${CFG_REL:-}" || -n "${LIG:-}" ]]; do
  # Skip blank or comment lines
  CFG_REL="$(printf '%s' "${CFG_REL:-}" | strip_cr)"
  LIG="$(printf '%s' "${LIG:-}" | strip_cr)"
  [[ -z "$CFG_REL" && -z "$LIG" ]] && continue
  [[ "$CFG_REL" =~ ^[[:space:]]*# ]] && continue

  GPU_ID=$(( gpu_toggle % 2 ))
  process_ligand "$CFG_REL" "$LIG" "$GPU_ID"
  ((gpu_toggle++))

  # Limit to avoid too many background jobs at once
  [[ $((gpu_toggle % 8)) -eq 0 ]] && wait
done < "$CHUNK_FILE"

wait

echo "-----------------------------------------------------"
echo "Finished chunk: $CHUNK_FILE"