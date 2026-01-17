#!/bin/bash
#
# array_job.sh - SLURM array job wrapper for virtual screening pipeline
#
# This script is submitted by the orchestrator and runs as each array task.
# It invokes the Python worker with the appropriate chunk ID.
#
# Usage (called by sbatch, not directly):
#   sbatch --array=0-N array_job.sh <stage> <chunk_dir> <results_dir> [config_path]
#
# Arguments:
#   $1 - stage: Pipeline stage name (e.g., docking, conversion)
#   $2 - chunk_dir: Directory containing chunk files
#   $3 - results_dir: Directory for result files
#   $4 - config_path: Path to config.yaml (optional)
#

set -euo pipefail

# Arguments
STAGE="${1:?Stage name required}"
CHUNK_DIR="${2:?Chunk directory required}"
RESULTS_DIR="${3:?Results directory required}"
CONFIG_PATH="${4:-config/config.yaml}"

# SLURM environment
TASK_ID="${SLURM_ARRAY_TASK_ID:?Must be run as SLURM array job}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-unknown}"

echo "========================================"
echo "SLURM Array Task"
echo "========================================"
echo "Job ID: ${JOB_ID}"
echo "Task ID: ${TASK_ID}"
echo "Stage: ${STAGE}"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "========================================"

# Load required modules (ARC guidance: load Anaconda, then activate)
if [[ -z "${PS1-}" ]]; then
    PS1=""
fi
module purge || true
ANACONDA_MODULE="${ANACONDA_MODULE:-Anaconda3/2020.11}"
module load "${ANACONDA_MODULE}" || module load Anaconda3 || true
module load Boost/1.77.0-GCC-11.2.0 CUDA/12.0.0 || true

# Activate the conda environment from $DATA to avoid $HOME
SNAKEMAKE_ENV="${SNAKEMAKE_CONDA_ENV:-snakemake_env}"
SNAKEMAKE_PREFIX="${SNAKEMAKE_CONDA_PREFIX:-}"
if [[ -z "$SNAKEMAKE_PREFIX" ]]; then
    if [[ "$SNAKEMAKE_ENV" == /* ]]; then
        SNAKEMAKE_PREFIX="$SNAKEMAKE_ENV"
    else
        SNAKEMAKE_PREFIX="${DATA:?DATA not set}/$SNAKEMAKE_ENV"
    fi
fi

# ARC guidance: prefer "source activate" in batch scripts
# shellcheck source=/dev/null
source activate "$SNAKEMAKE_PREFIX"

echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV:-none}"
echo "========================================"

# Change to project root
cd "${SLURM_SUBMIT_DIR:-.}"

# Run worker
python -m workflow.slurm.run \
    --stage "${STAGE}" \
    --worker \
    --chunk-id "${TASK_ID}" \
    --chunk-dir "${CHUNK_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --config "${CONFIG_PATH}"

EXIT_CODE=$?

echo "========================================"
echo "Task ${TASK_ID} finished with exit code ${EXIT_CODE}"
echo "Time: $(date)"
echo "========================================"

exit ${EXIT_CODE}
