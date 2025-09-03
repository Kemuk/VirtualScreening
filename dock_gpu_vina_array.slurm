#!/bin/bash
#SBATCH --job-name=litpcba_vina_gpu_array
#SBATCH --output=litpcba_vina_gpu_array_%A_%a.out
#SBATCH --error=litpcba_vina_gpu_array_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --clusters=htc
#SBATCH --gres=gpu:1
#SBATCH --array=0-13                  # 15 proteins => indices 0..14
# #SBATCH --constraint='gpu_sku:V100'
# #SBATCH -p short

set -euo pipefail

#################### PATHS (relative to this folder = project_root) ####################
DOCK_SRC="${DOCK_SRC:-$SLURM_SUBMIT_DIR/vina-gpu-dev}"     # your repo with Makefile
DATA_ROOT="${DATA_ROOT:-$SLURM_SUBMIT_DIR/LIT_PCBA}"       # your dataset
BOX_CSV="${BOX_CSV:-$DATA_ROOT/vina_boxes.csv}"
OUT_ROOT="${OUT_ROOT:-$SLURM_SUBMIT_DIR/LIT_PCBA_RESULTS}" # outputs created here
#######################################################################################

#################### Vina settings ####################################################
EXHAUSTIVENESS="${EXHAUSTIVENESS:-8}"
NUM_MODES="${NUM_MODES:-9}"
SEED="${SEED:-42}"                              # "" to let Vina choose
GPU_DEVICE_FLAG="${GPU_DEVICE_FLAG:---device 0}"# empty if your binary doesn't use it
RECEPTOR_GLOB="${RECEPTOR_GLOB:-*_protein.pdbqt}"
RESUME="${RESUME:-1}"                           # 1=skip ligands already done
BUILD="${BUILD:-1}"                             # 1=run make, 0=skip
DOCK_BIN="${DOCK_BIN:-}"                        # leave blank to auto-detect after build
#######################################################################################

module purge || true
module load Boost/1.77.0-GCC-11.2.0 || true
module load CUDA/12.0.0 || module load CUDA/11.8.0 || true   # OpenCL ICD via CUDA

# --- Build (once per task; incremental is fast) ---
if [[ "$BUILD" == "1" ]]; then
  echo "[BUILD] make -C $DOCK_SRC"
  [[ -d "$DOCK_SRC" ]] || { echo "ERROR: $DOCK_SRC not found"; exit 1; }
  make -C "$DOCK_SRC" -j "$(nproc)" || { echo "ERROR: make failed"; exit 1; }
fi

# Resolve executable if not provided
if [[ -z "$DOCK_BIN" ]]; then
  for cand in \
      "$DOCK_SRC/QuickVina2-GPU-2-1" \
      "$DOCK_SRC/vina-gpu" \
      "$DOCK_SRC/main/vina-gpu"; do
    [[ -x "$cand" ]] && DOCK_BIN="$cand" && break
  done
  [[ -n "$DOCK_BIN" ]] || DOCK_BIN="$(find "$DOCK_SRC" -maxdepth 1 -type f -perm -u+x \
                                      \( -iname '*vina*' -o -iname '*quickvina*' \) | head -n1 || true)"
fi
[[ -n "$DOCK_BIN" && -x "$DOCK_BIN" ]] || { echo "ERROR: GPU Vina executable not found; set DOCK_BIN"; exit 1; }
echo "[EXEC] Using $DOCK_BIN"

# --- helpers ---
get_box_for_protein () {
  awk -F',' -v target="$1" 'NR==1{next} $1==target{print $3","$4","$5","$6","$7","$8; exit}' "$BOX_CSV"
}
write_header_if_missing () {
  [[ -s "$1" ]] || echo "ligand,vina_score_kcal_mol,seed,exhaustiveness,num_modes,pose_file,log_file" > "$1"
}
extract_best_score () {
  local log="$1" pose="$2" s=""
  [[ -s "$log" ]] && s=$(awk '/REMARK VINA RESULT/{print $4; exit}' "$log" || true)
  [[ -z "$s" && -s "$pose" ]] && s=$(awk '/REMARK VINA RESULT/{print $4; exit}' "$pose" || true)
  echo "${s:-NA}"
}

# Protein list from boxes.csv (unique)
mapfile -t PROTEINS < <(awk -F',' 'NR>1{print $1}' "$BOX_CSV" | sort -u)
N=${#PROTEINS[@]}
(( N>0 )) || { echo "No proteins in $BOX_CSV"; exit 2; }

IDX=${SLURM_ARRAY_TASK_ID:?Missing SLURM_ARRAY_TASK_ID}
(( IDX>=0 && IDX<N )) || { echo "Array index ${IDX} out of range 0..$((N-1))"; exit 3; }
PROT="${PROTEINS[$IDX]}"

echo "[TASK $IDX] $PROT"
PROT_DIR="$DATA_ROOT/$PROT"
[[ -d "$PROT_DIR" ]] || { echo "WARN: $PROT_DIR missing"; exit 0; }

IFS=',' read -r CX CY CZ SX SY SZ < <(get_box_for_protein "$PROT")
[[ -n "${CX:-}" ]] || { echo "WARN: no box for $PROT"; exit 0; }

RECEPTOR_FILE="$(compgen -G "$PROT_DIR/$RECEPTOR_GLOB" | head -n1 || true)"
[[ -n "$RECEPTOR_FILE" ]] || { echo "WARN: receptor not found in $PROT_DIR"; exit 0; }

ACT_DIR="$PROT_DIR/pdbqt/actives"
INA_DIR="$PROT_DIR/pdbqt/inactives"

ROOT_P="$OUT_ROOT/$PROT"
META_DIR="$ROOT_P/meta"
D_ACT="$ROOT_P/docking/actives"
D_INA="$ROOT_P/docking/inactives"
S_DIR="$ROOT_P/docking/summaries"
mkdir -p "$META_DIR" "$D_ACT" "$D_INA" "$S_DIR"

# Metadata
if [[ ! -s "$META_DIR/run_meta.txt" ]]; then
  {
    echo "date: $(date -Iseconds)"
    echo "vina_bin: $DOCK_BIN"
    echo "exhaustiveness: $EXHAUSTIVENESS"
    echo "num_modes: $NUM_MODES"
    echo "seed: ${SEED:-auto}"
    echo "center: $CX,$CY,$CZ"
    echo "size: $SX,$SY,$SZ"
    echo "receptor: $RECEPTOR_FILE"
    "$DOCK_BIN" --version 2>&1 || true
  } > "$META_DIR/run_meta.txt"
fi
[[ -s "$META_DIR/box.csv" ]] || awk -F',' -v p="$PROT" 'NR==1||$1==p{print}' "$BOX_CSV" > "$META_DIR/box.csv"

SUM_ACT="$S_DIR/vina_scores_actives.csv"
SUM_INA="$S_DIR/vina_scores_inactives.csv"
write_header_if_missing "$SUM_ACT"
write_header_if_missing "$SUM_INA"

dock_folder () {
  local SRC_DIR="$1" ; local DST_DIR="$2" ; local SUM_CSV="$3"
  [[ -d "$SRC_DIR" ]] || { echo "NOTE: no dir $SRC_DIR"; return 0; }
  shopt -s nullglob
  for L in "$SRC_DIR"/*.pdbqt; do
    local STEM="${L##*/}"; STEM="${STEM%.pdbqt}"
    local OUT_POSE="$DST_DIR/${STEM}.pdbqt.out"
    local OUT_LOG="$DST_DIR/${STEM}.log"
    local OUT_SCORE_TXT="$DST_DIR/${STEM}.score.txt"

    if [[ "$RESUME" == "1" && -s "${OUT_POSE}.gz" ]]; then
      echo "SKIP (resume): ${OUT_POSE}.gz"
      continue
    fi

    # absolute paths so we can run from DOCK_SRC (kernels via relative paths)
    local ABS_REC ABS_LIG ABS_POSE ABS_LOG
    ABS_REC="$(readlink -f "$RECEPTOR_FILE")"
    ABS_LIG="$(readlink -f "$L")"
    mkdir -p "$DST_DIR"; ABS_POSE="$(readlink -f "$OUT_POSE" || true)"; ABS_LOG="$(readlink -f "$OUT_LOG" || true)"
    # if readlink -f fails before file exists, derive via pwd
    [[ -n "$ABS_POSE" ]] || ABS_POSE="$(cd "$DST_DIR" && pwd)/${STEM}.pdbqt.out"
    [[ -n "$ABS_LOG"  ]] || ABS_LOG="$(cd "$DST_DIR" && pwd)/${STEM}.log"

    CMD=( "$DOCK_BIN"
      --receptor "$ABS_REC"
      --ligand "$ABS_LIG"
      --center_x "$CX" --center_y "$CY" --center_z "$CZ"
      --size_x "$SX"  --size_y "$SY"  --size_z "$SZ"
      --exhaustiveness "$EXHAUSTIVENESS"
      --num_modes "$NUM_MODES"
      --out "$ABS_POSE"
      --log "$ABS_LOG"
    )
    [[ -n "${SEED:-}" ]] && CMD+=( --seed "$SEED" )
    [[ -n "${GPU_DEVICE_FLAG:-}" ]] && CMD+=( $GPU_DEVICE_FLAG )

    echo "Running in $(basename "$DOCK_SRC"): ${CMD[*]}"
    ( cd "$DOCK_SRC" && "${CMD[@]}" )

    local BEST; BEST="$(extract_best_score "$OUT_LOG" "$OUT_POSE")"
    echo "$BEST" > "$OUT_SCORE_TXT"

    gzip -f "$OUT_POSE" "$OUT_LOG"
    echo "${STEM},${BEST},${SEED:-auto},${EXHAUSTIVENESS},${NUM_MODES},${OUT_POSE}.gz,${OUT_LOG}.gz" >> "$SUM_CSV"
  done
}

dock_folder "$ACT_DIR" "$D_ACT" "$SUM_ACT"
dock_folder "$INA_DIR" "$D_INA" "$SUM_INA"

echo "Done: $PROT ? $ROOT_P"