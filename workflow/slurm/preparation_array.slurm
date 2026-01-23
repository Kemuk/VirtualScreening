#!/bin/bash
#SBATCH --job-name=preparation-array
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CHUNKS_DIR="${CHUNKS_DIR:-${PROJECT_DIR}/data/chunks/preparation}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/data/results/preparation}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/config/config.yaml}"

chunk_id="${SLURM_ARRAY_TASK_ID}"
chunk_path="${CHUNKS_DIR}/chunk_${chunk_id}.csv"
output_path="${RESULTS_DIR}/chunk_${chunk_id}.csv"

if [[ ! -f "$chunk_path" ]]; then
    echo "Chunk not found: ${chunk_path}"
    exit 0
fi

python workflow/scripts/process_stage_chunk.py \
    --stage preparation \
    --chunk "$chunk_path" \
    --output "$output_path" \
    --config "$CONFIG_PATH"
