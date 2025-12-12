#!/bin/bash
#SBATCH -J vina_smoke
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2        # 1 master + 1 GPU worker
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH -t 00:10:00
#SBATCH -o logs/%x-%A.out
#SBATCH -e logs/%x-%A.err

set -euo pipefail
ulimit -s 8192 || true
module purge || true
module load Anaconda3 Boost/1.77.0-GCC-11.2.0 CUDA/12.0.0 OpenMPI/4.1.1-GCC-11.2.0
cd "$SLURM_SUBMIT_DIR"
source activate ../vscreen_env
mkdir -p logs

CSV="LIT_PCBA/vina_boxes.csv"
VINA_BIN="vina-gpu-dev/QuickVina2-GPU-2-1"
SEED="0"
THREADS="0"

# Fallbacks so this works without an array:
TASK=${SLURM_ARRAY_TASK_ID:-0}
SHARDS=$(( ${SLURM_ARRAY_TASK_MAX:-0} + 1 ))
SHARD_DIR="${SLURM_TMPDIR:-/tmp}/vina_shard_${SLURM_JOB_ID:-smoke}_${TASK}"
mkdir -p "$SHARD_DIR"
SHARD_CSV="${SHARD_DIR}/vina_boxes_${TASK}_of_${SHARDS}.csv"
{ head -n 1 "$CSV"; tail -n +2 "$CSV" | awk -v m="$SHARDS" -v i="$TASK" ' (NR-1)%m==i '; } > "$SHARD_CSV"

echo "Smoke shard -> $(($(wc -l < "$SHARD_CSV")-1)) rows"
nvidia-smi -L || true

mpirun -np "$SLURM_NTASKS" \
  --map-by ppr:${SLURM_NTASKS_PER_NODE}:node --bind-to none \
  python run_vina_gpu_mpi.py \
    --csv "$SHARD_CSV" \
    --vina_bin "$VINA_BIN" \
    --seed "$SEED" \
    --threads "$THREADS" \
    --smoke --smoke_n 8 \
    --progress
