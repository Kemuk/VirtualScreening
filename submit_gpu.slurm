#!/bin/bash
#SBATCH --clusters=htc
#SBATCH -J vina_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --array=0-99
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --mail-user=reub0582@ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --requeue

set -euo pipefail
set -x

ulimit -s unlimited || true

module purge || true
module load Anaconda3 || true
module load Boost/1.77.0-GCC-11.2.0 CUDA/12.0.0 || true

# ensure we start in the submit directory
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# ensure logs exist for any runtime writes from the python script
mkdir -p logs

# activate conda environment robustly
conda activate ../snakemake_env

export PYTHONUNBUFFERED=1

# paths (relative to submit dir). Keep them quoted.
PYTHON_SCRIPT_DIR="$(cd "$(dirname "./dock.py")" && pwd)"
PYTHON_SCRIPT="$PYTHON_SCRIPT_DIR/$(basename "./dock.py")"

CSV_DIR="$(cd "$(dirname "./LIT_PCBA/vina_boxes.csv")" && pwd)"
CSV="$CSV_DIR/$(basename "./LIT_PCBA/vina_boxes.csv")"

MASTER_MANIFEST_DIR="$(cd "$(dirname "./LIT_PCBA/master_manifest.csv")" && pwd)"
MASTER_MANIFEST="$MASTER_MANIFEST_DIR/$(basename "./LIT_PCBA/master_manifest.csv")"

VINA_BIN_DIR="$(cd "$(dirname "./vina-gpu-dev/QuickVina2-GPU-2-1")" && pwd)"
VINA_BIN="$VINA_BIN_DIR/$(basename "./vina-gpu-dev/QuickVina2-GPU-2-1")"

# runtime variables with safe defaults

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"


GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

# runtime workers default to threads or 1 for GPU mode
RUNTIME_WORKERS="${RUNTIME_WORKERS:-$SLURM_CPUS_PER_TASK}"

echo "Using master manifest: $MASTER_MANIFEST"
echo "CSV: $CSV"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "Using vina binary: $VINA_BIN"
echo "Vina bin dir: $VINA_BIN_DIR"
echo "Running task id: $TASK_ID"
echo "GPU id: $GPU_ID"
echo "Runtime workers: $RUNTIME_WORKERS"

# switch working directory to the vina binary directory as requested
cd "$VINA_BIN_DIR"

# run
python -u "$PYTHON_SCRIPT" \
  --master_manifest "$MASTER_MANIFEST" \
  --task_id "$TASK_ID" \
  --mode gpu \
  --vina_bin "$VINA_BIN" \
  --gpu_id "$GPU_ID" \
  --runtime_workers "$RUNTIME_WORKERS"
