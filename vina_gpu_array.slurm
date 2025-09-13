#!/bin/bash
# file: vina_gpu_array.slurm

#SBATCH --job-name=vina_gpu
#SBATCH --output=logs/vina_gpu_%A_%a.out
#SBATCH --error=logs/vina_gpu_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --clusters=htc
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G

set -euo pipefail
ulimit -s 8192 || true

module purge || true
# Note: Ensure the Boost version here matches what your Vina binary was compiled with.
module load Boost/1.77.0-GCC-11.2.0 CUDA/12.0.0 || true

export CUDA_VISIBLE_DEVICES=0

# --- Helper Functions ---
trim(){ sed -E 's/^[ \t]+|[ \t]+$//g'; }
kv(){ awk -F'=' -v k="$1" '$1 ~ k {print $2}' "$2" | trim; }
# ---

cd "${SLURM_SUBMIT_DIR:?}"

QVINA_BIN="${QVINA_BIN:-"./vina-gpu-dev/QuickVina2-GPU-2-1"}"
[[ -x "$QVINA_BIN" ]] || { echo "ERROR: Vina-GPU not found at $(pwd)/$QVINA_BIN"; exit 1; }
BINDIR="$(cd "$(dirname "$QVINA_BIN")" && pwd -P)"

DATA_DIR="./LIT_PCBA"
CHUNKS_DIR="$DATA_DIR/chunks"

CHUNK_FILE=$(find "$CHUNKS_DIR" -maxdepth 1 -type f -name 'chunk_all_*.txt' | sort | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
[[ -f "$CHUNK_FILE" ]] || { echo "ERROR: Chunk file not found for task ID $SLURM_ARRAY_TASK_ID in $CHUNKS_DIR"; exit 1; }

echo "SLURM Job ID: ${SLURM_JOB_ID}, Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Working Directory: $(pwd)"
echo "Processing chunk: $CHUNK_FILE"
echo "Vina binary: $QVINA_BIN (run from $BINDIR)"
echo "-----------------------------------------------------"

# --- Main Processing Loop ---
ok=0; fail=0
last_cfg=""

while IFS=$'\t' read -r CFG_REL LIG; do
    CFG="$DATA_DIR/${CFG_REL#./}"

    [[ -f "$CFG" && -f "$LIG" ]] || { echo "SKIP: Missing file. CFG: $CFG, LIG: $LIG"; ((fail++)); continue; }

    if [[ "$CFG" != "$last_cfg" ]]; then
        echo "Parsing new config: $CFG"
        
        CFG_DIR="$(cd "$(dirname "$CFG")" && pwd -P)"
        
        RECEPTOR_REL=$(kv '^receptor' "$CFG")
        OUT_REL=$(kv '^out' "$CFG")
        CX=$(kv '^center_x' "$CFG"); CY=$(kv '^center_y' "$CFG"); CZ=$(kv '^center_z' "$CFG")
        SX=$(kv '^size_x' "$CFG");   SY=$(kv '^size_y' "$CFG");   SZ=$(kv '^size_z' "$CFG")
        THREAD_VAL=$(kv '^thread' "$CFG" || echo "8000")

        RECEPTOR="$CFG_DIR/${RECEPTOR_REL#./}"
        OUT_DIR="$CFG_DIR/${OUT_REL#./}"
        
        # **MODIFIED: Define a separate log directory and create both**
        LOG_DIR="$OUT_DIR/log"
        mkdir -p "$OUT_DIR" "$LOG_DIR"

        [[ -s "$RECEPTOR" ]] || { echo "FATAL: Receptor not found at path: $RECEPTOR (from config $CFG)"; exit 1; }
        last_cfg="$CFG"
    fi

    name="$(basename "$LIG")"
    out_pdbqt="$OUT_DIR/$name"
    # **MODIFIED: Use the new LOG_DIR for the log file path**
    log="$LOG_DIR/${name%.pdbqt}.log"

    set +e
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
    set -e

    if [[ $rc -ne 0 ]] || grep -qi "segmentation fault" "$log" || [[ ! -s "$out_pdbqt" ]]; then
        echo "[FAIL] $(basename "$CFG") :: $name (rc=$rc) -> $log"
        ((fail++))
    else
        echo "[OK]   $(basename "$CFG") :: $name"
        ((ok++))
    fi
done < "$CHUNK_FILE"

echo "-----------------------------------------------------"
echo "Chunk summary: OK=$ok, FAIL=$fail (File=$CHUNK_FILE)"
if (( fail > 0 )); then
    exit 1
fi
exit 0