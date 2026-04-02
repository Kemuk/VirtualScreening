#!/bin/bash
#SBATCH --job-name=ligand_based
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=data/logs/ligand_based/slurm_%j.out
#SBATCH --error=data/logs/ligand_based/slurm_%j.err

# Ligand-based screening (all targets, single SLURM job).
#
# Override METHOD at submission time:
#   sbatch --export=ALL,METHOD=usr workflow/slurm/ligand_based.slurm
#
# For a quick devel test on one target:
#   sbatch -p devel -t 00:10:00 --mem=4G --cpus-per-task=4 \
#       --wrap "python -m workflow.slurm.workers.ligand_based \
#           --manifest data/master/manifest.parquet \
#           --output   data/master/manifest_ligand_based_devel.parquet \
#           --target   ADRB2 \
#           --config   config/config.yaml"

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

mkdir -p data/logs/ligand_based

python -m workflow.slurm.workers.ligand_based \
    --manifest  data/master/manifest.parquet \
    --output    data/master/manifest_ligand_based.parquet \
    --work-dir  data/ligand_based \
    --method    "${METHOD:-usrcat}" \
    --config    config/config.yaml
