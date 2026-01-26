#!/bin/bash
#SBATCH --job-name=vina-backfill
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CHUNKS_DIR="${CHUNKS_DIR:-${PROJECT_DIR}/data/chunks/vina_backfill}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/data/results/vina_backfill}"
WORKERS="${WORKERS:-8}"

chunk_id="${SLURM_ARRAY_TASK_ID}"
chunk_path="${CHUNKS_DIR}/chunk_${chunk_id}.csv"
output_path="${RESULTS_DIR}/chunk_${chunk_id}.csv"

if [[ ! -f "$chunk_path" ]]; then
    echo "Chunk not found: ${chunk_path}"
    exit 0
fi

python workflow/scripts/backfill_vina_chunk.py \
    --chunk "$chunk_path" \
    --output "$output_path" \
    --workers "$WORKERS"
