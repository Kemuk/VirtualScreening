#!/bin/bash
#SBATCH --job-name=prepare-aev-plig-array
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CHUNKS_DIR="${CHUNKS_DIR:-${PROJECT_DIR}/data/chunks/rescoring}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/AEV-PLIG/data/shards}"

chunk_id="${SLURM_ARRAY_TASK_ID}"
chunk_path="${CHUNKS_DIR}/chunk_${chunk_id}.csv"
output_path="${OUTPUT_DIR}/lit_pcba_shard_${chunk_id}.csv"

if [[ ! -f "$chunk_path" ]]; then
    echo "Chunk not found: ${chunk_path}"
    exit 0
fi

python workflow/scripts/prepare_aev_plig_shard.py \
    --chunk "$chunk_path" \
    --output "$output_path"
