#!/bin/bash
#SBATCH --clusters=arc
#SBATCH -J vina_cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-99
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --mail-user=reub0582@ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --requeue

set -euo pipefail
set -x

# ===== ENV =====
ulimit -s 8192 || true
module purge || true
module load Anaconda3 || true
# optionally module load anything your code depends on, e.g:
module load AutoDock-Vina/1.2.3-foss-2021b || true

cd "$SLURM_SUBMIT_DIR"
source activate ../vscreen_env      # adjust this path to your environment
mkdir -p logs

# avoid oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ---- User config ----
CSV="LIT_PCBA/vina_boxes.csv"
PYTHON_SCRIPT="dock.py"               # your unified script
CPU_PER_DOCK=8
SHARD_TOTAL="${SLURM_ARRAY_TASK_COUNT:-1}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"

echo "python: $(which python)"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "Running with CPU_PER_DOCK=${CPU_PER_DOCK}"

srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  python -u "$PYTHON_SCRIPT" \
    --csv "$CSV" \
    --vina_bin vina \
    --mode cpu \
    --cpu "$CPU_PER_DOCK" \
    --workers 0 \
    --shard_total "$SHARD_TOTAL" \
    --shard_idx "$SHARD_IDX" \
    --skip_existing
