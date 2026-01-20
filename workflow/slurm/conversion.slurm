#!/bin/bash
#SBATCH --job-name=conversion
#SBATCH --output=data/logs/slurm/conversion_%A_%a.out
#SBATCH --error=data/logs/slurm/conversion_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --partition=arc
#SBATCH --cpus-per-task=2

# Conversion worker SLURM script (PDBQT to SDF)
# Usage: sbatch --array=0-499 workflow/slurm/conversion.slurm

# Configuration - adjust these as needed
NUM_CHUNKS=${SLURM_ARRAY_TASK_COUNT:-500}
PENDING_FILE="data/master/pending/conversion.parquet"
CONFIG_FILE="config/config.yaml"
RESULTS_DIR="data/master/results"

# Create log directory
mkdir -p data/logs/slurm

# Load OpenBabel if needed
module load OpenBabel 2>/dev/null || true

# Activate conda environment
source activate snakemake_env 2>/dev/null || conda activate snakemake_env 2>/dev/null || true

# Print job info
echo "=========================================="
echo "Conversion Worker (PDBQT -> SDF)"
echo "=========================================="
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Num chunks: ${NUM_CHUNKS}"
echo "Pending file: ${PENDING_FILE}"
echo "Hostname: $(hostname)"
echo "=========================================="

# Change to project directory
cd "${SLURM_SUBMIT_DIR:-/home/user/VirtualScreening}"

# Run worker
python -m workflow.slurm.workers.conversion \
    --pending "${PENDING_FILE}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --num-chunks "${NUM_CHUNKS}" \
    --config "${CONFIG_FILE}" \
    --results-dir "${RESULTS_DIR}"

echo "Task ${SLURM_ARRAY_TASK_ID} complete"
