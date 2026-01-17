#!/bin/bash
#
# submit_pipeline.sh - Submit the virtual screening pipeline to SLURM
#
# This script submits the orchestrator as a SLURM job, which then
# submits and monitors array jobs for each pipeline stage.
#
# Usage:
#   ./workflow/slurm/submit_pipeline.sh              # Production mode
#   ./workflow/slurm/submit_pipeline.sh --devel      # Devel mode (10k items, 5s timeout)
#   ./workflow/slurm/submit_pipeline.sh --stage docking  # Single stage
#

set -euo pipefail

# Defaults
DEVEL=""
STAGE="all"
PARTITION="devel"
TIME="00:30:00"      # 30 minutes for orchestrator (it just submits/waits)
MEM="4G"
CONFIG="config/config.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --devel)
            DEVEL="--devel"
            PARTITION="devel"
            TIME="00:10:00"  # 10 minutes for devel orchestrator
            shift
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--devel] [--stage <stage>] [--partition <part>] [--time HH:MM:SS]"
            exit 1
            ;;
    esac
done

# Get project root (parent of workflow directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Create logs directory
mkdir -p "${PROJECT_ROOT}/data/slurm/logs"

echo "========================================"
echo "Virtual Screening Pipeline Submission"
echo "========================================"
echo "Project root: ${PROJECT_ROOT}"
echo "Stage: ${STAGE}"
echo "Partition: ${PARTITION}"
echo "Time limit: ${TIME}"
echo "Devel mode: ${DEVEL:-no}"
echo "========================================"

# Submit orchestrator job
JOB_ID=$(sbatch \
    --partition="${PARTITION}" \
    --time="${TIME}" \
    --mem="${MEM}" \
    --job-name="vs-orchestrator" \
    --output="${PROJECT_ROOT}/data/slurm/logs/orchestrator_%j.out" \
    --error="${PROJECT_ROOT}/data/slurm/logs/orchestrator_%j.err" \
    --chdir="${PROJECT_ROOT}" \
    --parsable \
    --wrap="python -m workflow.slurm.run --stage ${STAGE} ${DEVEL} --config ${CONFIG}")

echo ""
echo "Submitted orchestrator job: ${JOB_ID}"
echo ""
echo "Monitor with:"
echo "  squeue -j ${JOB_ID}"
echo "  tail -f ${PROJECT_ROOT}/data/slurm/logs/orchestrator_${JOB_ID}.out"
echo ""
echo "Cancel with:"
echo "  scancel ${JOB_ID}"
echo "========================================"
