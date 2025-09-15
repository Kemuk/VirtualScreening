#!/usr/bin/env bash

# file: vina_gpu_array.simplified.slurm
# Goal: read EVERY line in the chunk file (tab-separated CFG\tLIG),
#       robust to CRLF, blanks, comments, and missing trailing newline.

#SBATCH --job-name=vina_gpu
#SBATCH --output=logs/vina_gpu_%A_%a.out
#SBATCH --error=logs/vina_gpu_%A_%a.err
#SBATCH --time=00:45:00
#SBATCH --partition=short
#SBATCH --clusters=htc
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --requeue


# Strict, but avoid -e to prevent early loop aborts on expected non-zero statuses.
set -u -o pipefail

ulimit -s 8192 || true

module purge || true
# Ensure Boost version matches your Vina build.
module load Boost/1.77.0-GCC-11.2.0 CUDA/12.0.0 || true

export CUDA_VISIBLE_DEVICES=0

# --- Helpers ---
trim(){ sed -E 's/^[[:space:]]+|[[:space:]]+$//g'; }
# awk exits non-zero if key not found -> enables sane defaults via `||` in subshell.
kv(){
  local key="$1" file="$2"
  awk -v k="^"$key"[[:space:]]*$" -F'=' '
    BEGIN{found=0}
    $1 ~ k {v=$2; gsub(/^[[:space:]]+|[[:space:]]+$/, "", v); print v; found=1}
    END{exit(found?0:1)}
  ' "$file"
}
# Normalize CR at EOL.
strip_cr(){ sed 's/\r$//'; }

# --- Setup ---
cd "${SLURM_SUBMIT_DIR:?}"

QVINA_BIN="${QVINA_BIN:-"./vina-gpu-dev/QuickVina2-GPU-2-1"}"
[[ -x "$QVINA_BIN" ]] || { echo "ERROR: Vina-GPU not found at $(pwd -P)/$QVINA_BIN"; exit 1; }
BINDIR="$(cd "$(dirname "$QVINA_BIN")" && pwd -P)"

DATA_DIR="./LIT_PCBA"
CHUNKS_DIR="$DATA_DIR/chunks"

# Map array index -> Nth chunk file (sorted). SLURM_ARRAY_TASK_ID is 0-based; sed is 1-based.
CHUNK_FILE=$(find "$CHUNKS_DIR" -maxdepth 1 -type f -name 'chunk_all_*.txt' | sort | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
[[ -f "$CHUNK_FILE" ]] || { echo "ERROR: Chunk file not found for task $SLURM_ARRAY_TASK_ID in $CHUNKS_DIR"; exit 1; }

printf 'SLURM Job ID: %s, Array Task ID: %s\n' "${SLURM_JOB_ID:-na}" "${SLURM_ARRAY_TASK_ID:-na}"
echo   "Working Directory: $(pwd -P)"
echo   "Processing chunk: $CHUNK_FILE"
echo   "Vina binary: $QVINA_BIN (run from $BINDIR)"
echo   "-----------------------------------------------------"

ok=0; fail=0
last_cfg=""

# Open CHUNK_FILE on FD 9; robust read handles last line without trailing \n.
exec 9<"$CHUNK_FILE"
while IFS=$'\t' read -r CFG_REL LIG <&9 || [[ -n "${CFG_REL:-}" || -n "${LIG:-}" ]]; do
  # Skip blanks/comments; strip CR if present.
  CFG_REL="$(printf '%s' "${CFG_REL:-}" | strip_cr)"
  LIG="$(printf '%s' "${LIG:-}" | strip_cr)"
  [[ -z "$CFG_REL" && -z "$LIG" ]] && continue
  [[ "$CFG_REL" =~ ^[[:space:]]*# ]] && continue

  # Resolve paths.
  CFG="$DATA_DIR/${CFG_REL#./}"
  # LIG may already be absolute; if relative, keep as-is relative to CWD
  if [[ "$LIG" != /* ]]; then LIG="$LIG"; fi

  # Per-line existence check (non-fatal).
  if [[ ! -f "$CFG" || ! -f "$LIG" ]]; then
    echo "SKIP: Missing file. CFG: $CFG, LIG: $LIG"
    ((fail++))
    continue
  fi

  # Parse config once per CFG; create out/log dirs.
  if [[ "$CFG" != "$last_cfg" ]]; then
    echo "Parsing config: $CFG"
    CFG_DIR="$(cd "$(dirname "$CFG")" && pwd -P)"

    RECEPTOR_REL=$(kv 'receptor' "$CFG" | trim || true)
    OUT_REL=$(kv 'out' "$CFG" | trim || true)
    CX=$(kv 'center_x' "$CFG" | trim || true)
    CY=$(kv 'center_y' "$CFG" | trim || true)
    CZ=$(kv 'center_z' "$CFG" | trim || true)
    SX=$(kv 'size_x' "$CFG"   | trim || true)
    SY=$(kv 'size_y' "$CFG"   | trim || true)
    SZ=$(kv 'size_z' "$CFG"   | trim || true)
    THREAD_VAL=$(kv 'thread' "$CFG" | trim 2>/dev/null || echo "8")  # default 8 if unspecified

    RECEPTOR="$CFG_DIR/${RECEPTOR_REL#./}"
    OUT_DIR="$CFG_DIR/${OUT_REL#./}"

    LOG_DIR="$OUT_DIR/log"
    mkdir -p "$OUT_DIR" "$LOG_DIR"

    [[ -s "$RECEPTOR" ]] || { echo "FATAL: Receptor not found at: $RECEPTOR (cfg: $CFG)"; exit 1; }
    last_cfg="$CFG"
  fi

  name="$(basename "$LIG")"
  out_pdbqt="$OUT_DIR/$name"
  log="$LOG_DIR/${name%.pdbqt}.log"

  # Run docking; never let a single failure abort the loop.
  (
    cd "$BINDIR" && "./$(basename "$QVINA_BIN")" \
      --receptor "$RECEPTOR" \
      --ligand   "$LIG" \
      --center_x "$CX" --center_y "$CY" --center_z "$CZ" \
      --size_x   "$SX" --size_y   "$SY" --size_z   "$SZ" \
      --out      "$out_pdbqt" \
      --seed     42 \
      --thread   "$THREAD_VAL" \
    >"$log" 2>&1
  )
  rc=$?

  # Consider rc!=0 OR empty output as failure. We *do not* grep for messages (flaky across builds).
  if [[ $rc -ne 0 || ! -s "$out_pdbqt" ]]; then
    echo "[FAIL] $(basename "$CFG") :: $name (rc=$rc) -> $log"
    ((fail++))
  else
    echo "[OK]   $(basename "$CFG") :: $name"
    ((ok++))
  fi

done
exec 9<&-

echo "-----------------------------------------------------"
echo "Chunk summary: OK=$ok, FAIL=$fail (File=$CHUNK_FILE)"

exit $(( fail>0 ? 1 : 0 ))
