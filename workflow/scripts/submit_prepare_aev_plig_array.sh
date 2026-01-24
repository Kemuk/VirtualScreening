#!/bin/bash
set -euo pipefail

chunks_dir=""
output_dir=""
log_dir=""
slurm_log_dir=""
config_path=""
mode="production"

while [[ $# -gt 0 ]]; do
    case $1 in
        --chunks-dir)
            chunks_dir="$2"
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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$chunks_dir" || -z "$output_dir" || -z "$log_dir" || -z "$slurm_log_dir" ]]; then
    echo "Usage: $0 --chunks-dir DIR --output-dir DIR --log-dir DIR --slurm-log-dir DIR [--config PATH] [--mode MODE]"
    exit 1
fi

mkdir -p "$output_dir" "$log_dir" "$slurm_log_dir"

chunk_count=$(ls "${chunks_dir}"/chunk_*.csv 2>/dev/null | wc -l | tr -d ' ')
if [[ "$chunk_count" -eq 0 ]]; then
    echo "No rescoring chunks found in ${chunks_dir}; skipping array submission."
    exit 0
fi

array_end=$((chunk_count - 1))

if [[ -n "$config_path" && ! -r "$config_path" ]]; then
    echo "Config not readable: $config_path" >&2
    exit 1
fi

partition=""
if [[ "$mode" == "devel" ]]; then
    partition="--partition=devel"
fi

time_limit="00:20:00"
if [[ "$mode" == "devel" ]]; then
    time_limit="00:10:00"
fi

array_job_raw=$(sbatch --parsable --array=0-"$array_end" \
    --job-name=vs-prepare-aev-plig-array \
    --time="$time_limit" \
    --output="${slurm_log_dir}/prepare_aev_plig_array_%A_%a.out" \
    --error="${slurm_log_dir}/prepare_aev_plig_array_%A_%a.err" \
    --clusters=arc \
    $partition \
    --export=ALL,CHUNKS_DIR="${chunks_dir}",OUTPUT_DIR="${output_dir}",CONFIG_PATH="${config_path}" \
    workflow/slurm/prepare_aev_plig_array.slurm)

array_job_id="${array_job_raw%%;*}"

echo "Submitted AEV-PLIG prepare array job: ${array_job_id} (raw: ${array_job_raw})"

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
    shard_path="${output_dir}/lit_pcba_shard_${shard_id}.csv"
    if [[ ! -f "$shard_path" ]]; then
        echo "Missing shard output: ${shard_path}" >&2
        missing_outputs=$((missing_outputs + 1))
    fi
done

if [[ "$missing_outputs" -gt 0 ]]; then
    echo "Missing ${missing_outputs} AEV-PLIG shard output file(s)." >&2
    exit 1
fi
