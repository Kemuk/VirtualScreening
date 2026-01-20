#!/bin/bash
#SBATCH --job-name=ligands
#SBATCH --output=data/logs/slurm/ligands_%A_%a.out
#SBATCH --error=data/logs/slurm/ligands_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --partition=arc
#SBATCH --cpus-per-task=4

# Ligands worker SLURM script (SMILES to PDBQT conversion)
# Usage: sbatch --array=0-499 workflow/slurm/ligands.slurm

# Configuration - adjust these as needed
NUM_CHUNKS=${SLURM_ARRAY_TASK_COUNT:-500}
PENDING_FILE="data/master/pending/ligands.parquet"
CONFIG_FILE="config/config.yaml"
RESULTS_DIR="data/master/results"

# Create log directory
mkdir -p data/logs/slurm

# Activate conda environment
source activate snakemake_env 2>/dev/null || conda activate snakemake_env 2>/dev/null || true

# Print job info
echo "=========================================="
echo "Ligands Worker (SMILES -> PDBQT)"
echo "=========================================="
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Num chunks: ${NUM_CHUNKS}"
echo "Pending file: ${PENDING_FILE}"
echo "Hostname: $(hostname)"
echo "=========================================="

# Change to project directory
cd "${SLURM_SUBMIT_DIR:-/home/user/VirtualScreening}"

# Run worker
python -m workflow.slurm.workers.ligands \
    --pending "${PENDING_FILE}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --num-chunks "${NUM_CHUNKS}" \
    --config "${CONFIG_FILE}" \
    --results-dir "${RESULTS_DIR}"

echo "Task ${SLURM_ARRAY_TASK_ID} complete"
