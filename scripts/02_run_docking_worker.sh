#!/usr/bin/env bash
#SBATCH --job-name=Vina_Dock_Worker
#SBATCH --output=log/dock_worker_%A_%a.out
#SBATCH --error=log/dock_worker_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=short
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --requeue

set -euo pipefail
set -x
IFS=$'\n\t'

# Worker script reads VINA_THREADS from the environment, defaulting to 2048
VINA_THREADS="${VINA_THREADS:-2048}"

PREP_DIR="${SLURM_SUBMIT_DIR}/prepared_data"
VINA_GPU_DIR="${SLURM_SUBMIT_DIR}/vina-gpu-dev"
BIN_CACHE_DIR="${SLURM_SUBMIT_DIR}/.cache/vina-gpu"

ulimit -s 8192 || true; export CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2 VINA_SEED=42 CUDA_DEVICE_ORDER="PCI_BUS_ID"
module purge || true; module load Boost/1.77.0-GCC-11.2.0 || true; module load CUDA/12.0.0 || true
log_msg() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task ${SLURM_ARRAY_TASK_ID}] $*"; }
trap 'gzip -f ${SLURM_SUBMIT_DIR}/log/dock_worker_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.{out,err} 2>/dev/null || true' EXIT
log_msg "Worker starting on $(hostname)"; nvidia-smi || true

JOB_INFO=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${PREP_DIR}/job_manifest.txt")
[[ -n "$JOB_INFO" ]] || { log_msg "ERROR: Could not get job info for task ${SLURM_ARRAY_TASK_ID}"; exit 1; }

# Parse manifest, ignoring the placeholder thread count from the file
TARGET=$(echo "$JOB_INFO" | cut -d',' -f1)
CHUNK_ID=$(echo "$JOB_INFO" | cut -d',' -f2)
RECEPTOR_PDBQT="${SLURM_SUBMIT_DIR}/$(echo "$JOB_INFO" | cut -d',' -f3)"
LIGANDS_TAR_GZ="${SLURM_SUBMIT_DIR}/$(echo "$JOB_INFO" | cut -d',' -f4)"
CX=$(echo "$JOB_INFO" | cut -d',' -f5); CY=$(echo "$JOB_INFO" | cut -d',' -f6); CZ=$(echo "$JOB_INFO" | cut -d',' -f7)
SX=$(echo "$JOB_INFO" | cut -d',' -f8); SY=$(echo "$JOB_INFO" | cut -d',' -f9); SZ=$(echo "$JOB_INFO" | cut -d',' -f10)
TARGET_OUT_DIR="${SLURM_SUBMIT_DIR}/$(echo "$JOB_INFO" | cut -d',' -f11)"

log_msg "Assigned Target: ${TARGET}, Chunk: ${CHUNK_ID}, Vina Threads: ${VINA_THREADS}"

LOCAL_SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
SCRATCH_DIR="${LOCAL_SCRATCH}/vina_${TARGET}_${CHUNK_ID}"
mkdir -p "${SCRATCH_DIR}"; trap 'rm -rf "${SCRATCH_DIR}"' EXIT

cp -f "$RECEPTOR_PDBQT" "${SCRATCH_DIR}/receptor.pdbqt"
cp -f "$LIGANDS_TAR_GZ" "${SCRATCH_DIR}/ligands.tar.gz"
tar -xzf "${SCRATCH_DIR}/ligands.tar.gz" -C "${SCRATCH_DIR}"
cp -f "${PREP_DIR}/${TARGET}/chunks/chunk_${CHUNK_ID}.txt" "${SCRATCH_DIR}/chunk_list.txt"

mkdir -p "$BIN_CACHE_DIR"; BIN="${BIN_CACHE_DIR}/QuickVina2-GPU-2-1"
if [[ ! -x "$BIN" ]]; then ( flock -w 900 200 || exit 1; if [[ ! -x "$BIN" ]]; then
make -C "$VINA_GPU_DIR" source; cp -f "$VINA_GPU_DIR/QuickVina2-GPU-2-1" "$BIN"; chmod +x "$BIN"; fi
) 200>"$BIN_CACHE_DIR/build.lock"; fi

CONFIG_FILE="${SCRATCH_DIR}/config.txt"; cat > "$CONFIG_FILE" <<-EOF
receptor = ${SCRATCH_DIR}/receptor.pdbqt
ligand_list = ${SCRATCH_DIR}/chunk_list.txt
base_path = ${SCRATCH_DIR}/ligands
center_x = ${CX}; center_y = ${CY}; center_z = ${CZ}
size_x = ${SX}; size_y = ${SY}; size_z = ${SZ}
thread = ${VINA_THREADS}
EOF

"${BIN}" --config "$CONFIG_FILE"

OUTPUT_PDBQT_DIR="${SCRATCH_DIR}/ligands_out"; SUMMARY_CSV="${SCRATCH_DIR}/summary_${TARGET}_${CHUNK_ID}.csv"
echo "ligand,score" > "$SUMMARY_CSV"
if [ -d "$OUTPUT_PDBQT_DIR" ] && [ "$(ls -A $OUTPUT_PDBQT_DIR)" ]; then
    grep -H "REMARK VINA RESULT:" "${OUTPUT_PDBQT_DIR}"/*.pdbqt | \
    awk -F '[:/ ]+' '{ligand=$2; sub(/_out\.pdbqt/, "", ligand); score=$(NF-2); print ligand","score}' >> "$SUMMARY_CSV"
fi

cp -f "$SUMMARY_CSV" "${TARGET_OUT_DIR}/"
log_msg "Worker finished."