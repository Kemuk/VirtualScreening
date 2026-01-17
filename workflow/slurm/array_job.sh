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

# Load required modules
module load Anaconda3 || true
module load Boost/1.77.0-GCC-11.2.0 CUDA/12.0.0 || true

# Initialize conda - try multiple methods
# Method 1: Use conda.sh from common locations
CONDA_PATHS=(
    "${HOME}/miniconda3/etc/profile.d/conda.sh"
    "${HOME}/anaconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
    "/apps/system/easybuild/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh"
)

CONDA_INITIALIZED=false
for conda_sh in "${CONDA_PATHS[@]}"; do
    if [[ -f "$conda_sh" ]]; then
        # shellcheck source=/dev/null
        source "$conda_sh"
        if type conda 2>/dev/null | grep -q "function"; then
            CONDA_INITIALIZED=true
            break
        fi
    fi
done

# Method 2: If conda command exists, use shell hook
if [[ "$CONDA_INITIALIZED" != "true" ]] && command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    if type conda 2>/dev/null | grep -q "function"; then
        CONDA_INITIALIZED=true
    fi
fi

# Method 3: Fallback to conda init if still not initialized
if [[ "$CONDA_INITIALIZED" != "true" ]] && command -v conda &> /dev/null; then
    conda init bash >/dev/null 2>&1 || true
    if [[ -f "${HOME}/.bashrc" ]]; then
        # shellcheck source=/dev/null
        source "${HOME}/.bashrc" 2>/dev/null || true
    fi
    if type conda 2>/dev/null | grep -q "function"; then
        CONDA_INITIALIZED=true
    fi
fi

# Activate the snakemake environment
SNAKEMAKE_ENV="${SNAKEMAKE_CONDA_ENV:-snakemake_env}"
if [[ "$CONDA_INITIALIZED" == "true" ]]; then
    conda activate "$SNAKEMAKE_ENV" 2>/dev/null || \
    conda activate /data/stat-cadd/reub0582/snakemake_env 2>/dev/null || \
    conda activate base 2>/dev/null || true
fi

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
