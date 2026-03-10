#!/bin/bash
#SBATCH --job-name=pipeline_orchestrator
#SBATCH --output=data/logs/slurm/orchestrator_%j.out
#SBATCH --error=data/logs/slurm/orchestrator_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --clusters=arc
#SBATCH --partition=short

# Pipeline Orchestrator
# Submits and monitors all pipeline stages from a compute node
#
# Usage (from login node):
#   sbatch workflow/slurm/orchestrator.slurm
#   sbatch --export=ALL,STAGES=docking,DEVEL=true workflow/slurm/orchestrator.slurm
#
# Environment variables (optional):
#   STAGES - Comma-separated stages (default: ligands,docking,conversion,aev_infer)
#   DEVEL - Set to "true" for devel mode (default: false)
#   MAX_ITEMS - Max items to process (default: unlimited for prod, 1000 for devel)
#   CHUNKS - Number of chunks (default: 500 for prod, 5 for devel)

set -euo pipefail
set -x

ulimit -s unlimited || true

# Load required modules
module purge || true
module load Anaconda3 || true

# Project directory
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_DIR}"

# Resolve conda environment
if [[ -n "${DATA:-}" ]]; then
    CONPREFIX="$(readlink -f "${DATA}/snakemake_env")"
else
    CONPREFIX="$(readlink -f "${PROJECT_DIR}/../snakemake_env")"
fi
PYTHON_BIN="${CONPREFIX}/bin/python"

export PYTHONUNBUFFERED=1

# Configuration from environment or defaults
STAGES="${STAGES:-ligands,docking,conversion,aev_infer}"
DEVEL="${DEVEL:-false}"
CHUNKS="${CHUNKS:-}"
MAX_ITEMS="${MAX_ITEMS:-}"

# Devel settings
DEVEL_CHUNKS=5
DEVEL_MAX_ITEMS=1000
DEVEL_PARTITION="devel"
DEVEL_TIME="00:10:00"

# Production settings
PROD_CHUNKS=500

# Stage-specific settings
declare -A STAGE_CLUSTER
STAGE_CLUSTER[ligands]="arc"
STAGE_CLUSTER[docking]="htc"
STAGE_CLUSTER[conversion]="arc"
STAGE_CLUSTER[aev_infer]="htc"

declare -A STAGE_TIME
STAGE_TIME[ligands]="01:00:00"
STAGE_TIME[docking]="02:00:00"
STAGE_TIME[conversion]="01:00:00"
STAGE_TIME[aev_infer]="02:00:00"

# Set settings based on mode
if [ "$DEVEL" = "true" ]; then
    CHUNKS="${CHUNKS:-$DEVEL_CHUNKS}"
    MAX_ITEMS="${MAX_ITEMS:-$DEVEL_MAX_ITEMS}"
    echo "=== DEVEL MODE ==="
    echo "  Max items: $MAX_ITEMS"
    echo "  Chunks: $CHUNKS"
else
    CHUNKS="${CHUNKS:-$PROD_CHUNKS}"
    echo "=== PRODUCTION MODE ==="
    echo "  Chunks: $CHUNKS"
    [ -n "$MAX_ITEMS" ] && echo "  Max items: $MAX_ITEMS"
fi

echo "Project directory: ${PROJECT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Stages: ${STAGES}"
echo ""

# Create log directory
LOG_DIR="${PROJECT_DIR}/data/logs/slurm"
mkdir -p "${LOG_DIR}"

# Convert stages to array
IFS=',' read -ra STAGE_ARRAY <<< "$STAGES"

# Process each stage sequentially
for STAGE in "${STAGE_ARRAY[@]}"; do
    echo "=========================================="
    echo "STAGE: $STAGE"
    echo "Started at: $(date)"
    echo "=========================================="

    # Build prepare command
    PREPARE_CMD="${PYTHON_BIN} -m workflow.slurm.prepare_stage --stage $STAGE --num-chunks $CHUNKS"
    PREPARE_CMD="$PREPARE_CMD --manifest ${PROJECT_DIR}/data/master/manifest.parquet"
    PREPARE_CMD="$PREPARE_CMD --output-dir ${PROJECT_DIR}/data/master"
    [ -n "$MAX_ITEMS" ] && PREPARE_CMD="$PREPARE_CMD --max-items $MAX_ITEMS"

    echo "Running: $PREPARE_CMD"
    PREPARE_OUTPUT=$($PREPARE_CMD 2>&1)
    echo "$PREPARE_OUTPUT"

    # Check if nothing to do
    if echo "$PREPARE_OUTPUT" | grep -q "Nothing to do"; then
        echo "Skipping $STAGE - already complete"
        echo ""
        continue
    fi

    # Extract actual chunk count
    ACTUAL_CHUNKS=$(echo "$PREPARE_OUTPUT" | grep "Actual chunks:" | awk '{print $3}')
    if [ -z "$ACTUAL_CHUNKS" ]; then
        echo "ERROR: Could not determine chunk count"
        exit 1
    fi

    # Build sbatch command
    ARRAY_END=$((ACTUAL_CHUNKS - 1))
    SBATCH_CMD="sbatch --parsable --array=0-$ARRAY_END%100"
    SBATCH_CMD="$SBATCH_CMD --output=${LOG_DIR}/${STAGE}_%A_%a.out"
    SBATCH_CMD="$SBATCH_CMD --error=${LOG_DIR}/${STAGE}_%A_%a.err"
    SBATCH_CMD="$SBATCH_CMD --export=ALL,PROJECT_DIR=${PROJECT_DIR},NUM_CHUNKS=${ACTUAL_CHUNKS}"

    if [ "$DEVEL" = "true" ]; then
        SBATCH_CMD="$SBATCH_CMD --clusters=${STAGE_CLUSTER[$STAGE]} --partition=$DEVEL_PARTITION --time=$DEVEL_TIME"
    else
        SBATCH_CMD="$SBATCH_CMD --clusters=${STAGE_CLUSTER[$STAGE]} --time=${STAGE_TIME[$STAGE]}"
    fi

    SBATCH_CMD="$SBATCH_CMD ${PROJECT_DIR}/workflow/slurm/${STAGE}.slurm"

    echo ""
    echo "Submitting: $SBATCH_CMD"

    # Submit and extract job ID
    JOB_RAW=$($SBATCH_CMD)
    JOB_ID="${JOB_RAW%%;*}"  # Remove ;cluster suffix

    if [ -z "$JOB_ID" ]; then
        echo "ERROR: Could not extract job ID"
        exit 1
    fi

    echo "Submitted job: $JOB_ID"
    echo ""
    echo "Waiting for job $JOB_ID to complete..."

    # Poll until job completes - query specific cluster to avoid multi-line output issues
    while true; do
        RUNNING=$(squeue -j "$JOB_ID" -h --clusters=${STAGE_CLUSTER[$STAGE]} 2>/dev/null | wc -l || echo "0")

        if [ "$RUNNING" -eq 0 ]; then
            echo "Job $JOB_ID completed at $(date)"
            break
        fi

        echo "  $(date +%H:%M:%S): $RUNNING tasks still running..."
        sleep 30
    done

    # Run update_manifest
    echo ""
    echo "Running update_manifest for $STAGE..."
    "${PYTHON_BIN}" -m workflow.slurm.update_manifest --stage "$STAGE" \
        --manifest "${PROJECT_DIR}/data/master/manifest.parquet" \
        --results-dir "${PROJECT_DIR}/data/master/results"

    echo ""
    echo "Stage $STAGE complete at $(date)"
    echo ""
done

echo "=========================================="
echo "ALL STAGES COMPLETE"
echo "Finished at: $(date)"
echo "=========================================="
