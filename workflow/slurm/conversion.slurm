#!/bin/bash
#SBATCH --job-name=conversion
#SBATCH --output=data/logs/slurm/conversion_%A_%a.out
#SBATCH --error=data/logs/slurm/conversion_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --clusters=arc
#SBATCH --cpus-per-task=2

# Conversion worker SLURM script (PDBQT to SDF)
# Usage: sbatch --array=0-499 workflow/slurm/conversion.slurm
#
# Note: run_stage.sh passes PROJECT_DIR and overrides --output/--error paths

set -euo pipefail

module purge

# Project directory (must be set by run_stage.sh or fallback to SLURM_SUBMIT_DIR)
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_DIR}"

# Configuration - all paths relative to PROJECT_DIR
NUM_CHUNKS=${NUM_CHUNKS:-${SLURM_ARRAY_TASK_COUNT:-500}}
PENDING_FILE="${PROJECT_DIR}/data/master/pending/conversion.parquet"
CONFIG_FILE="${PROJECT_DIR}/config/config.yaml"
RESULTS_DIR="${PROJECT_DIR}/data/master/results"

# Create log directory
mkdir -p "${PROJECT_DIR}/data/logs/slurm"

source $(conda info --base)/etc/profile.d/conda.sh

# Activate conda environment
conda activate "$DATA/snakemake_env"
export PATH="$DATA/snakemake_env/bin:$PATH"

export OBABEL_BIN="/data/stat-cadd/reub0582/snakemake_env/bin/obabel"
export BABEL_LIBDIR="/data/stat-cadd/reub0582/snakemake_env/lib/openbabel/3.1.0"
echo "BABEL_LIBDIR: ${BABEL_LIBDIR}"

which obabel

# Print job info
echo "=========================================="
echo "Conversion Worker (PDBQT -> SDF)"
echo "=========================================="
echo "Project dir: ${PROJECT_DIR}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Num chunks: ${NUM_CHUNKS}"
echo "Pending file: ${PENDING_FILE}"
echo "Hostname: $(hostname)"
echo "=========================================="

# Run worker
conda run -p $CONPREFIX python -m workflow.slurm.workers.conversion \
    --pending "${PENDING_FILE}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --num-chunks "${NUM_CHUNKS}" \
    --config "${CONFIG_FILE}" \
    --results-dir "${RESULTS_DIR}"

echo "Task ${SLURM_ARRAY_TASK_ID} complete"
