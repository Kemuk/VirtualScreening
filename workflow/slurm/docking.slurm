#!/bin/bash
#SBATCH --job-name=docking
#SBATCH --output=data/logs/slurm/docking_%A_%a.out
#SBATCH --error=data/logs/slurm/docking_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=htc
#SBATCH --gres=gpu:1

# Docking worker SLURM script
# Usage: sbatch --array=0-499 workflow/slurm/docking.slurm
#
# Note: run_stage.sh passes PROJECT_DIR and overrides --output/--error paths

set -euo pipefail
set -x

ulimit -s unlimited || true

# Load required modules (matching submit_gpu.slurm)
module purge || true
module load Anaconda3 || true
module load Boost/1.77.0-GCC-11.2.0 CUDA/12.0.0 || true

# Project directory (must be set by run_stage.sh or fallback to SLURM_SUBMIT_DIR)
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_DIR}"

# Configuration - all paths relative to PROJECT_DIR
NUM_CHUNKS=${NUM_CHUNKS:-${SLURM_ARRAY_TASK_COUNT:-500}}
PENDING_FILE="${PROJECT_DIR}/data/master/pending/docking.parquet"
CONFIG_FILE="${PROJECT_DIR}/config/config.yaml"
RESULTS_DIR="${PROJECT_DIR}/data/master/results"

# Create log directory
mkdir -p "${PROJECT_DIR}/data/logs/slurm"

# Activate conda environment
conda activate snakemake_env

export PYTHONUNBUFFERED=1

# Print job info
echo "=========================================="
echo "Docking Worker"
echo "=========================================="
echo "Project dir: ${PROJECT_DIR}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Num chunks: ${NUM_CHUNKS}"
echo "Pending file: ${PENDING_FILE}"
echo "Hostname: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "=========================================="

# Run worker
python -m workflow.slurm.workers.docking \
    --pending "${PENDING_FILE}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --num-chunks "${NUM_CHUNKS}" \
    --config "${CONFIG_FILE}" \
    --results-dir "${RESULTS_DIR}"

echo "Task ${SLURM_ARRAY_TASK_ID} complete"
