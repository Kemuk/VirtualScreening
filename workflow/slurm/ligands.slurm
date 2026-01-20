#!/bin/bash
#SBATCH --job-name=ligands
#SBATCH --output=data/logs/slurm/ligands_%A_%a.out
#SBATCH --error=data/logs/slurm/ligands_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --partition=arc
#SBATCH --cpus-per-task=4

# Ligands worker SLURM script (SMILES to PDBQT conversion)
# Usage: sbatch --array=0-499 workflow/slurm/ligands.slurm
#
# Note: run_stage.sh passes PROJECT_DIR and overrides --output/--error paths

set -euo pipefail
set -x

# Project directory (must be set by run_stage.sh or fallback to SLURM_SUBMIT_DIR)
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_DIR}"

# Configuration - all paths relative to PROJECT_DIR
NUM_CHUNKS=${NUM_CHUNKS:-${SLURM_ARRAY_TASK_COUNT:-500}}
PENDING_FILE="${PROJECT_DIR}/data/master/pending/ligands.parquet"
CONFIG_FILE="${PROJECT_DIR}/config/config.yaml"
RESULTS_DIR="${PROJECT_DIR}/data/master/results"

# Create log directory
mkdir -p "${PROJECT_DIR}/data/logs/slurm"

# Activate conda environment
source activate snakemake_env 2>/dev/null || conda activate snakemake_env 2>/dev/null || true

# Print job info
echo "=========================================="
echo "Ligands Worker (SMILES -> PDBQT)"
echo "=========================================="
echo "Project dir: ${PROJECT_DIR}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Num chunks: ${NUM_CHUNKS}"
echo "Pending file: ${PENDING_FILE}"
echo "Hostname: $(hostname)"
echo "=========================================="

# Run worker
python -m workflow.slurm.workers.ligands \
    --pending "${PENDING_FILE}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --num-chunks "${NUM_CHUNKS}" \
    --config "${CONFIG_FILE}" \
    --results-dir "${RESULTS_DIR}"

echo "Task ${SLURM_ARRAY_TASK_ID} complete"
