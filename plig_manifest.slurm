#!/bin/bash
# FILE: plig_manifest.slurm
#SBATCH --job-name=plig_manifest
#SBATCH --output=logs/plig_manifest_%A.out
#SBATCH --error=logs/plig_manifest_%A.err
#SBATCH --clusters=arc
#SBATCH --time=00:20:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=reub0582@ox.ac.uk
#SBATCH --partition=short


set -euo pipefail
set -x

module purge || true
module load Anaconda3
conda activate /data/stat-cadd/reub0582/snakemake_env


python3 make_aev_plig.py --csv LIT_PCBA/vina_boxes.csv --out LIT_PCBA/aev_plig.csv --shard_total 100