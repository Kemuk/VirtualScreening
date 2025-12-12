#!/bin/bash
#SBATCH --job-name=aevplig
#SBATCH --output=logs/aevplig_%A_%a.out
#SBATCH --error=logs/aevplig_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

ROOT="${1:-$SLURM_SUBMIT_DIR/LIT_PCBA}"
mkdir -p logs

# Discover dataset CSVs deterministically
mapfile -t DATASETS < <(find "$ROOT" -mindepth 3 -maxdepth 4 -type f -name 'aev_plig_*_*.csv' | sort)
N=${#DATASETS[@]}
[[ "$N" -gt 0 ]] || { echo "No AEV-PLIG dataset CSVs found under $ROOT"; exit 1; }

IDX="${SLURM_ARRAY_TASK_ID:?Need --array=...}"
[[ "$IDX" -ge 0 && "$IDX" -lt "$N" ]] || { echo "Array index $IDX out of range 0..$((N-1))"; exit 1; }

IN_CSV="${DATASETS[$IDX]}"
PROT_DIR="$(dirname "$(dirname "$IN_CSV")")"        # .../<TARGET>/rescoring
TARGET="$(basename "$(dirname "$PROT_DIR")")"       # <TARGET> protein dir
BASENAME="$(basename "$IN_CSV" .csv)"               # aev_plig_<TARGET>_<MODE>
MODE="${BASENAME##*_}"                               # actives|inactives

OUT_DIR="$PROT_DIR/results"
OUT_PRED="$OUT_DIR/${BASENAME}_preds.csv"
mkdir -p "$OUT_DIR"

echo "Task $IDX/$((N-1))  TARGET=$TARGET  MODE=$MODE"
echo "Input : $IN_CSV"
echo "Output: $OUT_PRED"

# ---- Load env (edit to your cluster) ----
module purge || true
module load CUDA/12.0.0 || true
# module load your/python/module || conda activate aev-plig

# ---- AEV-PLIG inference command ----
# Set this to your actual entrypoint, e.g.:
#   export AEV_PLIG_CMD="python -m aev_plig.infer"
# or export before sbatch: AEV_PLIG_CMD="aev_plig_infer"
AEV_PLIG_CMD="${AEV_PLIG_CMD:-aev_plig_infer}"

# Typical flags; adjust to your tool’s CLI.
# Expecting it to read the CSV with columns: unique_id,pK,sdf_file,pdb_file
# and write a CSV with at least: unique_id,aev_plig_score
set -x
$AEV_PLIG_CMD \
  --input_csv  "$IN_CSV" \
  --output_csv "$OUT_PRED" \
  --device     "cuda:0" \
  --batch_size 256
set +x

[[ -s "$OUT_PRED" ]] || { echo "ERROR: No predictions written to $OUT_PRED"; exit 2; }
echo "AEV-PLIG done: $OUT_PRED"
