#!/bin/bash
#SBATCH --job-name=docking
#SBATCH --output=data/logs/slurm/docking_%A_%a.out
#SBATCH --error=data/logs/slurm/docking_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --partition=htc
#SBATCH --gres=gpu:1

# Docking worker SLURM script
# Usage: sbatch --array=0-499 workflow/slurm/docking.slurm

# Configuration - adjust these as needed
NUM_CHUNKS=${SLURM_ARRAY_TASK_COUNT:-500}
PENDING_FILE="data/master/pending/docking.parquet"
CONFIG_FILE="config/config.yaml"
RESULTS_DIR="data/master/results"

# Create log directory
mkdir -p data/logs/slurm

# Load required modules (adjust for your cluster)
module load Boost/1.79.0-GCC-11.3.0 2>/dev/null || true
module load CUDA/11.7.0 2>/dev/null || true

# Activate conda environment
source activate snakemake_env 2>/dev/null || conda activate snakemake_env 2>/dev/null || true

# Print job info
echo "=========================================="
echo "Docking Worker"
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
python -m workflow.slurm.workers.docking \
    --pending "${PENDING_FILE}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --num-chunks "${NUM_CHUNKS}" \
    --config "${CONFIG_FILE}" \
    --results-dir "${RESULTS_DIR}"

echo "Task ${SLURM_ARRAY_TASK_ID} complete"
