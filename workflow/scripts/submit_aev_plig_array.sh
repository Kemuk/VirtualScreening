#!/bin/bash
set -euo pipefail

shards_dir=""
output_dir=""
log_dir=""
slurm_log_dir=""
config_path=""
mode="production"
model_name="model_GATv2Net_ligsim90_fep_benchmark"
aev_plig_env=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --shards-dir)
            shards_dir="$2"
            shift 2
            ;;
        --output-dir)
            output_dir="$2"
            shift 2
            ;;
        --log-dir)
            log_dir="$2"
            shift 2
            ;;
        --slurm-log-dir)
            slurm_log_dir="$2"
            shift 2
            ;;
        --config)
            config_path="$2"
            shift 2
            ;;
        --mode)
            mode="$2"
            shift 2
            ;;
        --model)
            model_name="$2"
            shift 2
            ;;
        --aev-plig-env)
            aev_plig_env="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$shards_dir" || -z "$output_dir" || -z "$log_dir" || -z "$slurm_log_dir" ]]; then
    echo "Usage: $0 --shards-dir DIR --output-dir DIR --log-dir DIR --slurm-log-dir DIR [--config PATH] [--mode MODE] [--model NAME] [--aev-plig-env PATH]"
    exit 1
fi

mkdir -p "$output_dir" "$log_dir" "$slurm_log_dir"

shard_count=$(ls "${shards_dir}"/lit_pcba_shard_*.csv 2>/dev/null | wc -l | tr -d ' ')
if [[ "$shard_count" -eq 0 ]]; then
    echo "No AEV-PLIG shards found in ${shards_dir}; skipping array submission."
    exit 0
fi

array_end=$((shard_count - 1))

if [[ -n "$config_path" && ! -r "$config_path" ]]; then
    echo "Config not readable: $config_path" >&2
    exit 1
fi

partition=""
if [[ "$mode" == "devel" ]]; then
    partition="--partition=devel"
fi

if [[ -z "$aev_plig_env" ]]; then
    if [[ -n "${DATA:-}" ]]; then
        aev_plig_env="${DATA}/aev-plig"
    else
        aev_plig_env="/data/stat-cadd/reub0582/aev-plig"
    fi
fi

time_limit="02:00:00"
if [[ "$mode" == "devel" ]]; then
    time_limit="00:10:00"
fi

array_job_raw=$(sbatch --parsable --array=0-"$array_end" \
    --job-name=vs-aev-plig-array \
    --time="$time_limit" \
    --output="${slurm_log_dir}/aev_plig_array_%A_%a.out" \
    --error="${slurm_log_dir}/aev_plig_array_%A_%a.err" \
    --clusters=htc \
    --gres=gpu:1 \
    $partition \
    --export=ALL,SHARDS_DIR="${shards_dir}",OUTPUT_DIR="${output_dir}",MODEL_NAME="${model_name}",AEV_PLIG_ENV="${aev_plig_env}" \
    workflow/slurm/aev_plig_array.slurm)

array_job_id="${array_job_raw%%;*}"

echo "Submitted AEV-PLIG array job: ${array_job_id} (raw: ${array_job_raw})"

while squeue -j "$array_job_id" -h >/dev/null 2>&1; do
    sleep 10
done

failed_tasks=$(sacct -j "$array_job_id" --format=State --noheader --parsable2 2>/dev/null \
    | awk -F'|' '$1 != "" && $1 != "COMPLETED" && $1 !~ /^CANCELLED/ {count++} END {print count+0}')

if [[ "$failed_tasks" -gt 0 ]]; then
    echo "Array job ${array_job_id} completed with ${failed_tasks} failed task(s)." >&2
    exit 1
fi

echo "Array job ${array_job_id} completed successfully."

missing_outputs=0
for shard_id in $(seq 0 "${array_end}"); do
    shard_path="${output_dir}/shard_${shard_id}_predictions.csv"
    if [[ ! -f "$shard_path" ]]; then
        echo "Missing prediction output: ${shard_path}" >&2
        missing_outputs=$((missing_outputs + 1))
    fi
done

if [[ "$missing_outputs" -gt 0 ]]; then
    echo "Missing ${missing_outputs} AEV-PLIG prediction file(s)." >&2
    exit 1
fi
