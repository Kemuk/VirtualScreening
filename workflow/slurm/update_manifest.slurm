#!/bin/bash
#SBATCH --job-name=update_manifest
#SBATCH --output=data/logs/slurm/update_manifest_%j.out
#SBATCH --error=data/logs/slurm/update_manifest_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=short

# Update manifest SLURM script
# Runs after array job completes to merge results back into manifest
#
# Usage: sbatch --dependency=afterok:ARRAY_JOB_ID workflow/slurm/update_manifest.slurm
#
# Required environment variables (passed via --export):
#   PROJECT_DIR - Absolute path to project directory
#   STAGE - Stage name (ligands, docking, conversion, aev_infer)

set -euo pipefail
set -x

ulimit -s unlimited || true

# Load required modules
module purge || true
module load Anaconda3 || true

# Project directory (must be set by run_stage.sh)
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_DIR}"

# Validate STAGE is set
if [[ -z "${STAGE:-}" ]]; then
    echo "ERROR: STAGE environment variable not set"
    exit 1
fi

# Create log directory
mkdir -p "${PROJECT_DIR}/data/logs/slurm"

# Resolve conda environment to absolute path
if [[ -n "${DATA:-}" ]]; then
    CONPREFIX="$(readlink -f "${DATA}/snakemake_env")"
else
    CONPREFIX="$(readlink -f "${PROJECT_DIR}/../snakemake_env")"
fi
export CONPREFIX

echo "Conda environment: ${CONPREFIX}"
PYTHON_BIN="${CONPREFIX}/bin/python"
echo "Python binary: ${PYTHON_BIN}"

export PYTHONUNBUFFERED=1

# Print job info
echo "=========================================="
echo "Update Manifest"
echo "=========================================="
echo "Project dir: ${PROJECT_DIR}"
echo "Stage: ${STAGE}"
echo "Hostname: $(hostname)"
echo "=========================================="

# Run update_manifest
"${PYTHON_BIN}" -m workflow.slurm.update_manifest \
    --stage "${STAGE}" \
    --manifest "${PROJECT_DIR}/data/master/manifest.parquet" \
    --results-dir "${PROJECT_DIR}/data/master/results"

echo "Update manifest for ${STAGE} complete"
