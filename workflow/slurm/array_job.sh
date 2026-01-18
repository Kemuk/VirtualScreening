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

ulimit -s unlimited || true

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

# Load required modules (match submit_gpu.slurm)
module purge || true
module load Anaconda3 || true
module load Boost/1.77.0-GCC-11.2.0 CUDA/12.0.0 || true

# ensure we start in the submit directory
cd "${SLURM_SUBMIT_DIR:-.}"

# Activate the conda environment (match submit_gpu.slurm)
SNAKEMAKE_PREFIX="${SNAKEMAKE_CONDA_PREFIX:-${SLURM_SUBMIT_DIR:-.}/../snakemake_env}"
SNAKEMAKE_PREFIX="$(cd "${SNAKEMAKE_PREFIX}" && pwd)"
source activate "${SNAKEMAKE_PREFIX}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

OBABEL_BIN="${SNAKEMAKE_PREFIX}/bin/obabel"
export OBABEL_BIN
BABEL_LIBDIR="${SNAKEMAKE_PREFIX}/lib/openbabel"
if [[ -d "${BABEL_LIBDIR}" ]]; then
    for candidate in "${BABEL_LIBDIR}"/*; do
        if [[ -d "${candidate}" ]]; then
            BABEL_LIBDIR="${candidate}"
            break
        fi
    done
fi
export BABEL_LIBDIR
BABEL_DATADIR="${SNAKEMAKE_PREFIX}/share/openbabel"
if [[ -d "${BABEL_DATADIR}" ]]; then
    for candidate in "${BABEL_DATADIR}"/*; do
        if [[ -d "${candidate}" ]]; then
            BABEL_DATADIR="${candidate}"
            break
        fi
    done
fi
export BABEL_DATADIR

PYTHON_BIN="${SNAKEMAKE_PREFIX}/bin/python"
echo "Python: ${PYTHON_BIN}"
echo "Conda env: ${CONDA_DEFAULT_ENV:-none}"
echo "========================================"

# Change to project root
cd "${SLURM_SUBMIT_DIR:-.}"

# Run worker
"${PYTHON_BIN}" -m workflow.slurm.run \
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
