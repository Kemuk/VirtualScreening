#!/bin/bash
#SBATCH --job-name=aev-plig-array
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

module load Boost/1.77.0-GCC-11.2.0 CUDA/12.0.0 || true

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
AEV_PLIG_DIR="${PROJECT_DIR}/AEV-PLIG"
SHARDS_DIR="${SHARDS_DIR:-${AEV_PLIG_DIR}/data/shards}"
OUTPUT_DIR="${OUTPUT_DIR:-${AEV_PLIG_DIR}/output/shards}"
MODEL_NAME="${MODEL_NAME:-model_GATv2Net_ligsim90_fep_benchmark}"
if [[ -n "${AEV_PLIG_ENV:-}" ]]; then
    resolved_aev_plig_env="${AEV_PLIG_ENV}"
elif [[ -n "${DATA:-}" ]]; then
    resolved_aev_plig_env="${DATA}/aev-plig"
else
    resolved_aev_plig_env="/data/stat-cadd/reub0582/aev-plig"
fi

if [[ -f "${resolved_aev_plig_env}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${resolved_aev_plig_env}/bin/activate"
else
    echo "AEV-PLIG environment not found at ${resolved_aev_plig_env}" >&2
    exit 1
fi

shard_id="${SLURM_ARRAY_TASK_ID}"
shard_input="${SHARDS_DIR}/lit_pcba_shard_${shard_id}.csv"
shard_output="${OUTPUT_DIR}/shard_${shard_id}_predictions.csv"

if [[ ! -f "$shard_input" ]]; then
    echo "Shard not found: ${shard_input}"
    exit 0
fi

mkdir -p "$OUTPUT_DIR"

cd "$AEV_PLIG_DIR"

python process_and_predict.py \
    --dataset_csv="data/shards/lit_pcba_shard_${shard_id}.csv" \
    --data_name="shard_${shard_id}" \
    --trained_model_name="${MODEL_NAME}"

# Move output to expected location (AEV-PLIG may output to different paths)
if [[ -f "output/predictions/lit_pcba_shard_${shard_id}_predictions.csv" ]]; then
    mv "output/predictions/lit_pcba_shard_${shard_id}_predictions.csv" \
       "output/shards/shard_${shard_id}_predictions.csv"
elif [[ -f "output/predictions/shard_${shard_id}_predictions.csv" ]]; then
    mv "output/predictions/shard_${shard_id}_predictions.csv" \
       "output/shards/shard_${shard_id}_predictions.csv"
fi
