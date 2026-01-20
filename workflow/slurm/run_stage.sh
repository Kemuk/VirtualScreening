#!/bin/bash
#
# run_stage.sh - Submit pipeline stages with SLURM job dependencies
#
# Usage:
#   ./run_stage.sh                              # Run all stages (production)
#   ./run_stage.sh --stage docking             # Run specific stage
#   ./run_stage.sh --stage docking,conversion  # Run multiple stages
#   ./run_stage.sh --devel                     # Run with devel settings
#
# Jobs are chained with dependencies - you can disconnect after submission.
# Monitor with: squeue -u $USER
#

set -euo pipefail

# Get absolute project directory (where this script is run from)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_DIR

# Change to project directory
cd "${PROJECT_DIR}"

echo "Project directory: ${PROJECT_DIR}"

# Default settings
STAGES="ligands,docking,conversion,aev_infer"
DEVEL=false
PREPARE_ONLY=false
UPDATE_ONLY=false

# Production settings
PROD_CHUNKS=500
PROD_MAX_ITEMS=""  # unlimited

# Devel settings
DEVEL_CHUNKS=5
DEVEL_MAX_ITEMS=1000
DEVEL_PARTITION="devel"
DEVEL_TIME="00:10:00"

# Update manifest job settings
UPDATE_PARTITION="short"
UPDATE_TIME="00:30:00"

# Stage-specific settings
declare -A STAGE_PARTITION
STAGE_PARTITION[ligands]="arc"
STAGE_PARTITION[docking]="htc"
STAGE_PARTITION[conversion]="arc"
STAGE_PARTITION[aev_infer]="htc"

declare -A STAGE_TIME
STAGE_TIME[ligands]="01:00:00"
STAGE_TIME[docking]="02:00:00"
STAGE_TIME[conversion]="01:00:00"
STAGE_TIME[aev_infer]="02:00:00"

# Resolve conda environment for PYTHON_BIN
if [[ -n "${DATA:-}" ]]; then
    CONPREFIX="$(readlink -f "${DATA}/snakemake_env")"
else
    CONPREFIX="$(readlink -f "${PROJECT_DIR}/../snakemake_env")"
fi
PYTHON_BIN="${CONPREFIX}/bin/python"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            STAGES="$2"
            shift 2
            ;;
        --devel)
            DEVEL=true
            shift
            ;;
        --chunks)
            PROD_CHUNKS="$2"
            shift 2
            ;;
        --max-items)
            PROD_MAX_ITEMS="$2"
            shift 2
            ;;
        --prepare-only)
            PREPARE_ONLY=true
            shift
            ;;
        --update-only)
            UPDATE_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --stage STAGES     Comma-separated stages (default: all)"
            echo "                     Valid: ligands,docking,conversion,aev_infer"
            echo "  --devel            Use devel settings (1000 items, 5 chunks, devel partition)"
            echo "  --chunks N         Number of chunks (default: 500 for prod, 5 for devel)"
            echo "  --max-items N      Max items to process (default: unlimited for prod, 1000 for devel)"
            echo "  --prepare-only     Only run prepare_stage.py, don't submit"
            echo "  --update-only      Only run update_manifest.py (no job submission)"
            echo "  --help             Show this help"
            echo ""
            echo "Jobs are chained with dependencies. You can disconnect after submission."
            echo "Monitor with: squeue -u \$USER"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set settings based on mode
if [ "$DEVEL" = true ]; then
    CHUNKS=$DEVEL_CHUNKS
    MAX_ITEMS=$DEVEL_MAX_ITEMS
    UPDATE_PARTITION=$DEVEL_PARTITION
    UPDATE_TIME=$DEVEL_TIME
    echo "=== DEVEL MODE ==="
    echo "  Max items: $MAX_ITEMS"
    echo "  Chunks: $CHUNKS"
    echo "  Partition: $DEVEL_PARTITION"
    echo "  Time: $DEVEL_TIME"
else
    CHUNKS=$PROD_CHUNKS
    MAX_ITEMS=$PROD_MAX_ITEMS
    echo "=== PRODUCTION MODE ==="
    echo "  Chunks: $CHUNKS"
    [ -n "$MAX_ITEMS" ] && echo "  Max items: $MAX_ITEMS"
fi
echo ""

# Convert stages to array
IFS=',' read -ra STAGE_ARRAY <<< "$STAGES"

# Track submitted jobs for final summary
declare -a SUBMITTED_JOBS=()

# Track the last update job ID for chaining to next stage
LAST_UPDATE_JOB_ID=""

# Process each stage
for STAGE in "${STAGE_ARRAY[@]}"; do
    echo "=========================================="
    echo "STAGE: $STAGE"
    echo "=========================================="

    # Skip to update if --update-only
    if [ "$UPDATE_ONLY" = true ]; then
        echo "Running update_manifest.py..."
        "${PYTHON_BIN}" -m workflow.slurm.update_manifest --stage "$STAGE" \
            --manifest "${PROJECT_DIR}/data/master/manifest.parquet" \
            --results-dir "${PROJECT_DIR}/data/master/results"
        continue
    fi

    # Build prepare command with absolute paths
    PREPARE_CMD="${PYTHON_BIN} -m workflow.slurm.prepare_stage --stage $STAGE --num-chunks $CHUNKS"
    PREPARE_CMD="$PREPARE_CMD --manifest ${PROJECT_DIR}/data/master/manifest.parquet"
    PREPARE_CMD="$PREPARE_CMD --output-dir ${PROJECT_DIR}/data/master"
    [ -n "$MAX_ITEMS" ] && PREPARE_CMD="$PREPARE_CMD --max-items $MAX_ITEMS"

    echo "Running: $PREPARE_CMD"

    # Run prepare and capture output
    PREPARE_OUTPUT=$($PREPARE_CMD 2>&1)
    echo "$PREPARE_OUTPUT"

    # Check if nothing to do
    if echo "$PREPARE_OUTPUT" | grep -q "Nothing to do"; then
        echo "Skipping $STAGE - already complete"
        echo ""
        continue
    fi

    # Extract actual chunk count from output
    ACTUAL_CHUNKS=$(echo "$PREPARE_OUTPUT" | grep "Actual chunks:" | awk '{print $3}')
    if [ -z "$ACTUAL_CHUNKS" ]; then
        echo "ERROR: Could not determine chunk count"
        exit 1
    fi

    # Stop here if --prepare-only
    if [ "$PREPARE_ONLY" = true ]; then
        echo "Prepared $STAGE (--prepare-only, not submitting)"
        echo ""
        continue
    fi

    # Build sbatch command for array job
    ARRAY_END=$((ACTUAL_CHUNKS - 1))
    LOG_DIR="${PROJECT_DIR}/data/logs/slurm"
    mkdir -p "${LOG_DIR}"

    SBATCH_CMD="sbatch --parsable --array=0-$ARRAY_END"
    SBATCH_CMD="$SBATCH_CMD --output=${LOG_DIR}/${STAGE}_%A_%a.out"
    SBATCH_CMD="$SBATCH_CMD --error=${LOG_DIR}/${STAGE}_%A_%a.err"
    SBATCH_CMD="$SBATCH_CMD --export=ALL,PROJECT_DIR=${PROJECT_DIR},NUM_CHUNKS=${ACTUAL_CHUNKS}"

    # Add dependency on previous stage's update job if exists
    if [ -n "$LAST_UPDATE_JOB_ID" ]; then
        SBATCH_CMD="$SBATCH_CMD --dependency=afterok:${LAST_UPDATE_JOB_ID}"
    fi

    if [ "$DEVEL" = true ]; then
        SBATCH_CMD="$SBATCH_CMD --partition=$DEVEL_PARTITION --time=$DEVEL_TIME"
    else
        SBATCH_CMD="$SBATCH_CMD --partition=${STAGE_PARTITION[$STAGE]} --time=${STAGE_TIME[$STAGE]}"
    fi

    SBATCH_CMD="$SBATCH_CMD ${PROJECT_DIR}/workflow/slurm/${STAGE}.slurm"

    echo ""
    echo "Submitting array job: $SBATCH_CMD"

    # Submit array job and capture job ID
    ARRAY_JOB_ID=$($SBATCH_CMD)

    if [ -z "$ARRAY_JOB_ID" ]; then
        echo "ERROR: Could not extract array job ID"
        exit 1
    fi

    echo "  Array job ID: $ARRAY_JOB_ID"
    SUBMITTED_JOBS+=("${STAGE}_array:${ARRAY_JOB_ID}")

    # Submit update_manifest job with dependency on array job
    UPDATE_CMD="sbatch --parsable"
    UPDATE_CMD="$UPDATE_CMD --dependency=afterok:${ARRAY_JOB_ID}"
    UPDATE_CMD="$UPDATE_CMD --output=${LOG_DIR}/update_${STAGE}_%j.out"
    UPDATE_CMD="$UPDATE_CMD --error=${LOG_DIR}/update_${STAGE}_%j.err"
    UPDATE_CMD="$UPDATE_CMD --export=ALL,PROJECT_DIR=${PROJECT_DIR},STAGE=${STAGE}"
    UPDATE_CMD="$UPDATE_CMD --partition=${UPDATE_PARTITION} --time=${UPDATE_TIME}"
    UPDATE_CMD="$UPDATE_CMD ${PROJECT_DIR}/workflow/slurm/update_manifest.slurm"

    echo "Submitting update job: $UPDATE_CMD"

    UPDATE_JOB_ID=$($UPDATE_CMD)

    if [ -z "$UPDATE_JOB_ID" ]; then
        echo "ERROR: Could not extract update job ID"
        exit 1
    fi

    echo "  Update job ID: $UPDATE_JOB_ID (depends on $ARRAY_JOB_ID)"
    SUBMITTED_JOBS+=("${STAGE}_update:${UPDATE_JOB_ID}")

    # Save for next stage's dependency
    LAST_UPDATE_JOB_ID="$UPDATE_JOB_ID"

    echo ""
done

# Final summary
echo "=========================================="
echo "ALL JOBS SUBMITTED"
echo "=========================================="
echo ""
echo "Submitted jobs:"
for JOB in "${SUBMITTED_JOBS[@]}"; do
    echo "  $JOB"
done
echo ""
echo "You can now disconnect. Jobs will run automatically."
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed"
echo ""
echo "Check logs in: ${LOG_DIR}/"
echo "=========================================="
