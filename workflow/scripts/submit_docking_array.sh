#!/bin/bash
set -euo pipefail

chunks_dir=""
results_dir=""
log_dir=""
slurm_log_dir=""
config_path=""

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
        --config)
            config_path="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$chunks_dir" || -z "$results_dir" || -z "$log_dir" || -z "$slurm_log_dir" ]]; then
    echo "Usage: $0 --chunks-dir DIR --results-dir DIR --log-dir DIR --slurm-log-dir DIR [--config PATH]"
    exit 1
fi

mkdir -p "$results_dir" "$log_dir" "$slurm_log_dir"

chunk_count=$(ls "${chunks_dir}"/chunk_*.csv 2>/dev/null | wc -l | tr -d ' ')
if [[ "$chunk_count" -eq 0 ]]; then
    echo "No docking chunks found in ${chunks_dir}; skipping array submission."
    exit 0
fi

array_end=$((chunk_count - 1))

if [[ -n "$config_path" ]]; then
    if [[ ! -r "$config_path" ]]; then
        echo "Config not readable: $config_path" >&2
        exit 1
    fi
    docking_mode=$(python -c 'import sys,yaml; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get("docking", {}).get("mode", "cpu"))' "$config_path")
else
    docking_mode="cpu"
fi

cluster="arc"
gres=""
if [[ "$docking_mode" == "gpu" ]]; then
    cluster="htc"
    gres="--gres=gpu:1"
fi

partition=""
if [[ "${config_path:-}" != "" ]]; then
    mode=$(python -c 'import sys,yaml; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get("mode", "production"))' "$config_path")
    if [[ "$mode" == "devel" ]]; then
        partition="--partition=devel"
    fi
fi

array_job_raw=$(sbatch --parsable --array=0-"$array_end" \
    --job-name=vs-docking-array \
    --output="${slurm_log_dir}/docking_array_%A_%a.out" \
    --error="${slurm_log_dir}/docking_array_%A_%a.err" \
    --clusters="$cluster" \
    $partition \
    $gres \
    --export=ALL,CHUNKS_DIR="${chunks_dir}",RESULTS_DIR="${results_dir}",CONFIG_PATH="${config_path}" \
    workflow/slurm/docking_array.slurm)

array_job_id="${array_job_raw%%;*}"

echo "Submitted docking array job: ${array_job_id} (raw: ${array_job_raw})"

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
