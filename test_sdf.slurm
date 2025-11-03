#!/bin/bash
#SBATCH --clusters=arc
#SBATCH -J make_sdf
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-0
#SBATCH -t 00:10:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --partition=devel

set -euo pipefail
set -x

ulimit -s 8192 || true
module purge || true
module load Anaconda3 || true
module load Boost/1.77.0-GCC-11.2.0 || true

cd "$SLURM_SUBMIT_DIR"
source activate ../vscreen_env || true
mkdir -p logs

# avoid oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ---- User config ----
CSV="LIT_PCBA/vina_boxes.csv"
PYTHON_SCRIPT="make_sdf_files.py"
MASTER_MANIFEST="LIT_PCBA/sdf_manifest.csv"

# shard / task info
SHARD_TOTAL="${SLURM_ARRAY_TASK_COUNT:-1}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "python: $(which python)"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "Task id: $TASK_ID  Shard total: $SHARD_TOTAL"

python3 -u "$PYTHON_SCRIPT" \
  --csv "$CSV" \
  --master_manifest "$MASTER_MANIFEST" \
  --task_id "$TASK_ID" \
  --shard_total "$SHARD_TOTAL" \
  --obabel_bin obabel \
