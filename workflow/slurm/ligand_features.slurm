#!/bin/bash
#SBATCH --job-name=ligand_features
#SBATCH --output=data/logs/slurm/ligand_features_%A_%a.out
#SBATCH --error=data/logs/slurm/ligand_features_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --clusters=arc
#SBATCH --cpus-per-task=4

# Ligand features worker — Stage 1 of ligand-based screening.
# Generates per-ligand feature vectors (fingerprints or 3D shape descriptors).
#
# Usage:
#   sbatch --array=0-499 workflow/slurm/ligand_features.slurm
#
# After all array tasks complete, run the merge step:
#   python -m workflow.slurm.workers.ligand_features --merge \
#       --results-dir data/master/results/ligand_features \
#       --cache-dir   data/ligand_based/cache

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_DIR}"

NUM_CHUNKS=${NUM_CHUNKS:-${SLURM_ARRAY_TASK_COUNT:-500}}
METHOD="${METHOD:-usrcat}"
PENDING_FILE="${PROJECT_DIR}/data/master/pending/ligand_features.parquet"
CONFIG_FILE="${PROJECT_DIR}/config/config.yaml"
RESULTS_DIR="${PROJECT_DIR}/data/master/results/ligand_features"
CACHE_DIR="${PROJECT_DIR}/data/ligand_based/cache"

mkdir -p "${PROJECT_DIR}/data/logs/slurm"

source activate snakemake_env 2>/dev/null || conda activate snakemake_env 2>/dev/null || true

echo "=========================================="
echo "Ligand Features Worker (Stage 1)"
echo "=========================================="
echo "Project dir: ${PROJECT_DIR}"
echo "Task ID:     ${SLURM_ARRAY_TASK_ID}"
echo "Num chunks:  ${NUM_CHUNKS}"
echo "Method:      ${METHOD}"
echo "Pending:     ${PENDING_FILE}"
echo "Hostname:    $(hostname)"
echo "=========================================="

python -m workflow.slurm.workers.ligand_features \
    --pending     "${PENDING_FILE}" \
    --task-id     "${SLURM_ARRAY_TASK_ID}" \
    --num-chunks  "${NUM_CHUNKS}" \
    --config      "${CONFIG_FILE}" \
    --results-dir "${RESULTS_DIR}" \
    --cache-dir   "${CACHE_DIR}"

echo "Task ${SLURM_ARRAY_TASK_ID} complete"
