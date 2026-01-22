#!/bin/bash
#SBATCH --job-name=docking-array
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -euo pipefail

module load Boost/1.77.0-GCC-11.2.0 CUDA/12.0.0 || true

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CHUNKS_DIR="${CHUNKS_DIR:-${PROJECT_DIR}/data/chunks/docking}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/data/results/docking}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/config/config.yaml}"

chunk_id="${SLURM_ARRAY_TASK_ID}"
chunk_path="${CHUNKS_DIR}/chunk_${chunk_id}.csv"
output_path="${RESULTS_DIR}/chunk_${chunk_id}.csv"

if [[ ! -f "$chunk_path" ]]; then
    echo "Chunk not found: ${chunk_path}"
    exit 0
fi

python workflow/scripts/process_stage_chunk.py \
    --stage docking \
    --chunk "$chunk_path" \
    --output "$output_path" \
    --config "$CONFIG_PATH"
