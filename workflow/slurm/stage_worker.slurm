#!/bin/bash
#SBATCH --job-name=stage-worker
#SBATCH --output=data/logs/slurm/%x_%A_%a.out
#SBATCH --error=data/logs/slurm/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Generic stage worker SLURM script
#
# This template works for all pipeline stages (ligands, docking, conversion, aev_infer).
# The STAGE environment variable determines which worker module to run.
#
# Usage via submit_stage.py (recommended):
#   python -m workflow.slurm.submit_stage --stage docking
#
# Direct usage (advanced):
#   export STAGE=docking
#   export PROJECT_DIR=/path/to/project
#   export NUM_CHUNKS=500
#   sbatch --array=0-499 --export=ALL,STAGE,PROJECT_DIR,NUM_CHUNKS \
#          workflow/slurm/stage_worker.slurm
#
# Note: submit_stage.py overrides resource specifications (#SBATCH directives above)
# via command-line arguments based on stage_config.py settings.

set -euo pipefail

# Validate required environment variables
if [[ -z "${STAGE:-}" ]]; then
    echo "ERROR: STAGE environment variable not set"
    exit 1
fi

if [[ -z "${PROJECT_DIR:-}" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
fi

cd "${PROJECT_DIR}"

# Configuration - all paths relative to PROJECT_DIR
NUM_CHUNKS=${NUM_CHUNKS:-${SLURM_ARRAY_TASK_COUNT:-500}}
PENDING_FILE="${PROJECT_DIR}/data/master/pending/${STAGE}.parquet"
CONFIG_FILE="${PROJECT_DIR}/config/config.yaml"
RESULTS_DIR="${PROJECT_DIR}/data/master/results"

# Create log directory
mkdir -p "${PROJECT_DIR}/data/logs/slurm"

# Resolve conda environment to absolute path
if [[ -n "${DATA:-}" ]]; then
    CONPREFIX="$(readlink -f "${DATA}/snakemake_env")"
else
    CONPREFIX="$(readlink -f "${PROJECT_DIR}/../snakemake_env")"
fi
export CONPREFIX

PYTHON_BIN="${CONPREFIX}/bin/python"
export PYTHONUNBUFFERED=1

# Load stage-specific modules
case "${STAGE}" in
    docking)
        # GPU docking requires CUDA and Boost
        module purge 2>/dev/null || true
        module load Anaconda3 2>/dev/null || true
        module load Boost/1.77.0-GCC-11.2.0 CUDA/12.0.0 2>/dev/null || true
        set -x  # Enable command tracing for GPU jobs
        ulimit -s unlimited || true
        ;;
    conversion)
        # SDF conversion requires OpenBabel
        module load OpenBabel 2>/dev/null || true
        ;;
    aev_infer)
        # AEV-PLIG rescoring requires CUDA for neural network
        module purge 2>/dev/null || true
        module load Anaconda3 2>/dev/null || true
        module load CUDA/12.0.0 2>/dev/null || true
        ;;
    ligands)
        # Ligand preparation has minimal requirements
        :
        ;;
    *)
        echo "WARNING: Unknown stage '${STAGE}', proceeding without stage-specific modules"
        ;;
esac

# Print job info
echo "=========================================="
echo "Stage Worker: ${STAGE}"
echo "=========================================="
echo "Project dir: ${PROJECT_DIR}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Num chunks: ${NUM_CHUNKS}"
echo "Pending file: ${PENDING_FILE}"
echo "Results dir: ${RESULTS_DIR}"
echo "Hostname: $(hostname)"
echo "Python: ${PYTHON_BIN}"

# Print GPU info if available
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "GPU: ${CUDA_VISIBLE_DEVICES}"
fi

echo "=========================================="

# Validate pending file exists
if [[ ! -f "${PENDING_FILE}" ]]; then
    echo "ERROR: Pending file not found: ${PENDING_FILE}"
    echo "Run prepare_stage.py first to create pending parquet file"
    exit 1
fi

# Run stage-specific worker module
"${PYTHON_BIN}" -m "workflow.slurm.workers.${STAGE}" \
    --pending "${PENDING_FILE}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --num-chunks "${NUM_CHUNKS}" \
    --config "${CONFIG_FILE}" \
    --results-dir "${RESULTS_DIR}"

echo "Task ${SLURM_ARRAY_TASK_ID} complete"
