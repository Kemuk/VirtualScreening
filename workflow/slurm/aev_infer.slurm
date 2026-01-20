#!/bin/bash
#SBATCH --job-name=aev_infer
#SBATCH --output=data/logs/slurm/aev_infer_%A_%a.out
#SBATCH --error=data/logs/slurm/aev_infer_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --partition=htc
#SBATCH --gres=gpu:1

# AEV-PLIG inference worker SLURM script
# Usage: sbatch --array=0-99 workflow/slurm/aev_infer.slurm

# Configuration - adjust these as needed
NUM_CHUNKS=${SLURM_ARRAY_TASK_COUNT:-100}
PENDING_FILE="data/master/pending/aev_infer.parquet"
CONFIG_FILE="config/config.yaml"
RESULTS_DIR="data/master/results"

# Create log directory
mkdir -p data/logs/slurm

# Load required modules
module load CUDA/11.7.0 2>/dev/null || true

# Activate conda environment
source activate snakemake_env 2>/dev/null || conda activate snakemake_env 2>/dev/null || true

# Print job info
echo "=========================================="
echo "AEV-PLIG Inference Worker"
echo "=========================================="
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Num chunks: ${NUM_CHUNKS}"
echo "Pending file: ${PENDING_FILE}"
echo "Hostname: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "=========================================="

# Change to project directory
cd "${SLURM_SUBMIT_DIR:-/home/user/VirtualScreening}"

# Run worker
python -m workflow.slurm.workers.aev_infer \
    --pending "${PENDING_FILE}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --num-chunks "${NUM_CHUNKS}" \
    --config "${CONFIG_FILE}" \
    --results-dir "${RESULTS_DIR}"

echo "Task ${SLURM_ARRAY_TASK_ID} complete"
