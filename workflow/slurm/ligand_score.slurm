#!/bin/bash
#SBATCH --job-name=ligand_score
#SBATCH --output=data/logs/slurm/ligand_score_%A_%a.out
#SBATCH --error=data/logs/slurm/ligand_score_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --clusters=arc
#SBATCH --cpus-per-task=1

# Ligand score worker — Stage 2 of ligand-based screening.
# Scores each ligand against its target's template molecule.
#
# Prerequisites:
#   1. Stage 1 (ligand_features) must be complete and merged into features.parquet
#   2. Pending parquet must have template_smiles / template_source columns:
#      python -m workflow.slurm.workers.ligand_score --prepare \
#          --pending  data/master/pending/ligand_score.parquet \
#          --manifest data/master/manifest.parquet
#
# Usage:
#   sbatch --array=0-499 workflow/slurm/ligand_score.slurm
#
# After all array tasks complete, run the merge step:
#   python -m workflow.slurm.workers.ligand_score --merge \
#       --results-dir data/master/results/ligand_score \
#       --manifest    data/master/manifest.parquet \
#       --output      data/master/manifest_ligand_based.parquet

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_DIR}"

NUM_CHUNKS=${NUM_CHUNKS:-${SLURM_ARRAY_TASK_COUNT:-500}}
PENDING_FILE="${PROJECT_DIR}/data/master/pending/ligand_score.parquet"
CONFIG_FILE="${PROJECT_DIR}/config/config.yaml"
RESULTS_DIR="${PROJECT_DIR}/data/master/results"
CACHE_DIR="${PROJECT_DIR}/data/ligand_based/cache"

mkdir -p "${PROJECT_DIR}/data/logs/slurm"

source activate snakemake_env 2>/dev/null || conda activate snakemake_env 2>/dev/null || true

echo "=========================================="
echo "Ligand Score Worker (Stage 2)"
echo "=========================================="
echo "Project dir: ${PROJECT_DIR}"
echo "Task ID:     ${SLURM_ARRAY_TASK_ID}"
echo "Num chunks:  ${NUM_CHUNKS}"
echo "Pending:     ${PENDING_FILE}"
echo "Cache dir:   ${CACHE_DIR}"
echo "Hostname:    $(hostname)"
echo "=========================================="

python -m workflow.slurm.workers.ligand_score \
    --pending     "${PENDING_FILE}" \
    --task-id     "${SLURM_ARRAY_TASK_ID}" \
    --num-chunks  "${NUM_CHUNKS}" \
    --config      "${CONFIG_FILE}" \
    --results-dir "${RESULTS_DIR}" \
    --cache-dir   "${CACHE_DIR}"

echo "Task ${SLURM_ARRAY_TASK_ID} complete"
