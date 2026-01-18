#!/bin/bash
#
# submit_pipeline.sh - Submit the virtual screening pipeline to SLURM
#
# This script submits the orchestrator as a SLURM job, which then
# submits and monitors array jobs for each pipeline stage.
#
# Usage:
#   ./workflow/slurm/submit_pipeline.sh              # Production mode
#   ./workflow/slurm/submit_pipeline.sh --devel      # Devel mode (10k items, partition default time)
#   ./workflow/slurm/submit_pipeline.sh --stage docking  # Single stage
#

set -euo pipefail

# Defaults
DEVEL=""
STAGE="all"
CLUSTER="arc"
PARTITION=""
TIME="00:30:00"      # 30 minutes for orchestrator (it just submits/waits)
TIME_SET="false"
MEM="4G"
CONFIG="config/config.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --devel)
            DEVEL="--devel"
            PARTITION="devel"
            if [[ "${TIME_SET}" == "false" ]]; then
                TIME=""
            fi
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
            TIME_SET="true"
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
echo "Cluster: ${CLUSTER}"
echo "Partition: ${PARTITION:-none}"
echo "Time limit: ${TIME:-partition default}"
echo "Devel mode: ${DEVEL:-no}"
echo "========================================"

# Submit orchestrator job
PARTITION_FLAG=()
if [[ -n "${PARTITION}" ]]; then
    PARTITION_FLAG=(--partition="${PARTITION}")
fi
TIME_FLAG=()
if [[ -n "${TIME}" ]]; then
    TIME_FLAG=(--time="${TIME}")
fi

JOB_ID=$(sbatch \
    --clusters="${CLUSTER}" \
    "${PARTITION_FLAG[@]}" \
    "${TIME_FLAG[@]}" \
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
