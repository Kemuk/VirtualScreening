#!/bin/bash
set -euo pipefail

chunks_dir=""
results_dir=""
log_dir=""
slurm_log_dir=""
mode="production"
workers=8

while [[ $# -gt 0 ]]; do
    case $1 in
        --chunks-dir)
            chunks_dir="$2"
            shift 2
            ;;
        --results-dir)
            results_dir="$2"
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
        --mode)
            mode="$2"
            shift 2
            ;;
        --workers)
            workers="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$chunks_dir" || -z "$results_dir" || -z "$log_dir" || -z "$slurm_log_dir" ]]; then
    echo "Usage: $0 --chunks-dir DIR --results-dir DIR --log-dir DIR --slurm-log-dir DIR [--mode MODE] [--workers N]"
    exit 1
fi

mkdir -p "$results_dir" "$log_dir" "$slurm_log_dir"

chunk_count=$(ls "${chunks_dir}"/chunk_*.csv 2>/dev/null | wc -l | tr -d ' ')
if [[ "$chunk_count" -eq 0 ]]; then
    echo "No vina_backfill chunks found in ${chunks_dir}; skipping array submission."
    exit 0
fi

array_end=$((chunk_count - 1))

partition=""
if [[ "$mode" == "devel" ]]; then
    partition="--partition=devel"
fi

time_limit="00:15:00"
if [[ "$mode" == "devel" ]]; then
    time_limit="00:05:00"
fi

array_job_raw=$(sbatch --parsable --array=0-"$array_end" \
    --job-name=vs-vina-backfill \
    --time="$time_limit" \
    --output="${slurm_log_dir}/vina_backfill_%A_%a.out" \
    --error="${slurm_log_dir}/vina_backfill_%A_%a.err" \
    --clusters=arc \
    $partition \
    --export=ALL,CHUNKS_DIR="${chunks_dir}",RESULTS_DIR="${results_dir}",WORKERS="${workers}" \
    workflow/slurm/vina_backfill_array.slurm)

array_job_id="${array_job_raw%%;*}"

echo "Submitted vina_backfill array job: ${array_job_id} (raw: ${array_job_raw})"

while squeue --clusters=arc -j "$array_job_id" -h >/dev/null 2>&1; do
    sleep 10
done

failed_tasks=$(sacct --clusters=arc -j "$array_job_id" --format=State --noheader --parsable2 2>/dev/null \
    | awk -F'|' '$1 != "" && $1 != "COMPLETED" && $1 !~ /^CANCELLED/ {count++} END {print count+0}')

if [[ "$failed_tasks" -gt 0 ]]; then
    echo "Array job ${array_job_id} completed with ${failed_tasks} failed task(s)." >&2
    exit 1
fi

echo "Array job ${array_job_id} completed successfully."
