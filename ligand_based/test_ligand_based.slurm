#!/bin/bash
#SBATCH --clusters=arc
#SBATCH -J test_ligand_based
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --array=0-2
#SBATCH -t 00:10:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --mail-user=reub0582@ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --requeue
#SBATCH --partition=devel

# Make all job files local to the job working directory; avoid /tmp or cluster scratch
mkdir -p "$PWD/logs" "$PWD/output" "$PWD/tmp"
export TMPDIR="$PWD/tmp"
export TMP="$PWD/tmp"
export TEMP="$PWD/tmp"
export XDG_RUNTIME_DIR="$PWD/tmp"

set -euo pipefail
set -x

ulimit -s 8192 || true

module purge || true
module load Anaconda3 || true
conda config --set auto_activate_base false
eval "$(conda shell.bash hook)"
module load Boost/1.77.0-GCC-11.2.0 CUDA || true
conda activate /data/stat-cadd/reub0582/snakemake_env

# Use working directory (where you submit) not the spooled script location
SCRIPT_DIR="$PWD"
MANIFEST="${SCRIPT_DIR}/manifest_1k.csv"

ID0=${SLURM_ARRAY_TASK_ID}                     # 0-based
N=${SLURM_ARRAY_TASK_COUNT:-$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))}
FIFO="${SCRIPT_DIR}/fifo_${SLURM_JOB_ID}_${ID0}"
PART="${SCRIPT_DIR}/output/LIT_PCBA_${SLURM_JOB_ID}_${ID0}_predictions.csv"
MASTER="${SCRIPT_DIR}/output/LIT_PCBA_predictions.csv"

# ensure dirs exist (redundant safe)
mkdir -p "${SCRIPT_DIR}/logs" "${SCRIPT_DIR}/output" "${SCRIPT_DIR}/tmp"

# quick sanity: manifest exists
if [ ! -f "$MANIFEST" ]; then
  echo "manifest not found: $MANIFEST" >&2
  exit 1
fi

# Exit if this task has no rows
awk -v id0="$ID0" -v m="$N" 'NR>1 && ((NR-2)%m)==id0{c++} END{exit !(c>0)}' "$MANIFEST" \
  || { echo "Task $ID0: no rows"; exit 0; }

trap 'rm -f "$FIFO"' EXIT
rm -f "$FIFO" "$PART" || true
mkfifo "$FIFO"

# stream this task's slice (preserve header) into FIFO
awk -v id0="$ID0" -v m="$N" 'NR==1 || (NR>1 && ((NR-2)%m)==id0)' "$MANIFEST" > "$FIFO" & WRITER=$!

# Use Python from the activated conda env and emit diagnostics
PY_EXE="${CONDA_PREFIX:-/data/stat-cadd/reub0582/snakemake_env}/bin/python"
# print interpreter and oddt availability to the python stderr log for debugging
"$PY_EXE" - <<'PYDIAG' 1>&2 || true
import sys, pkgutil
print("PYTHON:", sys.executable)
spec = pkgutil.find_loader("oddt")
print("oddt_spec:", spec)
PYDIAG

# run worker with the same interpreter
"$PY_EXE" "${SCRIPT_DIR}/worker_stream.py" \
  --input-fifo "$FIFO" \
  --out-csv "$PART" \
  --n-jobs "${SLURM_CPUS_PER_TASK:-4}" \
  2>> "${SCRIPT_DIR}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}_${ID0}.py.err" \
  || { kill "$WRITER" 2>/dev/null || true; exit 1; }

wait "$WRITER" || true

# If PART missing or empty exit
[ -s "$PART" ] || { echo "Task $ID0: no output"; rm -f "$PART"; exit 0; }

# Atomically append PART to MASTER with flock (write header only once)
{
  flock -x 9
  if [ ! -s "$MASTER" ]; then
    head -n 1 "$PART" >&9
  fi
  tail -n +2 "$PART" >&9
} 9>>"$MASTER"

# cleanup
rm -f "$PART" "$FIFO"
