#!/bin/bash
#
# run_stage.sh - Convenience script for running pipeline stages
#
# Usage:
#   ./run_stage.sh                              # Run all stages (production)
#   ./run_stage.sh --stage docking             # Run specific stage
#   ./run_stage.sh --stage docking,conversion  # Run multiple stages
#   ./run_stage.sh --devel                     # Run with devel settings
#   ./run_stage.sh --stage docking --wait      # Wait for completion
#

set -euo pipefail
set -x

# Get absolute project directory (where this script is run from)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_DIR

# Change to project directory
cd "${PROJECT_DIR}"

echo "Project directory: ${PROJECT_DIR}"

# Default settings
STAGES="ligands,docking,conversion,aev_infer"
DEVEL=false
WAIT=false
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
        --wait)
            WAIT=true
            shift
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
            echo "  --wait             Wait for jobs to complete before moving to next stage"
            echo "  --prepare-only     Only run prepare_stage.py, don't submit"
            echo "  --update-only      Only run update_manifest.py"
            echo "  --help             Show this help"
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

# Process each stage
for STAGE in "${STAGE_ARRAY[@]}"; do
    echo "=========================================="
    echo "STAGE: $STAGE"
    echo "=========================================="

    # Skip to update if --update-only
    if [ "$UPDATE_ONLY" = true ]; then
        echo "Running update_manifest.py..."
        python -m workflow.slurm.update_manifest --stage "$STAGE" \
            --manifest "${PROJECT_DIR}/data/master/manifest.parquet" \
            --results-dir "${PROJECT_DIR}/data/master/results"
        continue
    fi

    # Build prepare command with absolute paths
    PREPARE_CMD="python -m workflow.slurm.prepare_stage --stage $STAGE --num-chunks $CHUNKS"
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

    # Build sbatch command with absolute paths
    ARRAY_END=$((ACTUAL_CHUNKS - 1))
    LOG_DIR="${PROJECT_DIR}/data/logs/slurm"
    mkdir -p "${LOG_DIR}"

    SBATCH_CMD="sbatch --array=0-$ARRAY_END"
    SBATCH_CMD="$SBATCH_CMD --output=${LOG_DIR}/${STAGE}_%A_%a.out"
    SBATCH_CMD="$SBATCH_CMD --error=${LOG_DIR}/${STAGE}_%A_%a.err"
    SBATCH_CMD="$SBATCH_CMD --export=ALL,PROJECT_DIR=${PROJECT_DIR},NUM_CHUNKS=${ACTUAL_CHUNKS}"

    if [ "$DEVEL" = true ]; then
        SBATCH_CMD="$SBATCH_CMD --partition=$DEVEL_PARTITION --time=$DEVEL_TIME"
    else
        SBATCH_CMD="$SBATCH_CMD --partition=${STAGE_PARTITION[$STAGE]} --time=${STAGE_TIME[$STAGE]}"
    fi

    SBATCH_CMD="$SBATCH_CMD ${PROJECT_DIR}/workflow/slurm/${STAGE}.slurm"

    echo ""
    echo "Submitting: $SBATCH_CMD"

    # Submit and capture job ID
    SUBMIT_OUTPUT=$($SBATCH_CMD)
    echo "$SUBMIT_OUTPUT"

    JOB_ID=$(echo "$SUBMIT_OUTPUT" | grep -oP 'Submitted batch job \K\d+')

    if [ -z "$JOB_ID" ]; then
        echo "ERROR: Could not extract job ID"
        exit 1
    fi

    echo "Job ID: $JOB_ID"

    # Wait for completion if requested
    if [ "$WAIT" = true ]; then
        echo ""
        echo "Waiting for job $JOB_ID to complete..."

        while true; do
            # Check if any jobs still running
            RUNNING=$(squeue -j "$JOB_ID" -h 2>/dev/null | wc -l)

            if [ "$RUNNING" -eq 0 ]; then
                echo "Job $JOB_ID completed"
                break
            fi

            echo "  $RUNNING tasks still running..."
            sleep 30
        done

        # Run update_manifest
        echo ""
        echo "Running update_manifest.py..."
        python -m workflow.slurm.update_manifest --stage "$STAGE" \
            --manifest "${PROJECT_DIR}/data/master/manifest.parquet" \
            --results-dir "${PROJECT_DIR}/data/master/results"
    else
        echo ""
        echo "Job submitted. Run these commands after completion:"
        echo "  squeue -j $JOB_ID                           # check status"
        echo "  python -m workflow.slurm.update_manifest --stage $STAGE --manifest ${PROJECT_DIR}/data/master/manifest.parquet --results-dir ${PROJECT_DIR}/data/master/results  # update manifest"
    fi

    echo ""
done

echo "=========================================="
echo "Done!"
echo "=========================================="
