#!/usr/bin/env bash
#SBATCH --job-name=vina_cpu_array
#SBATCH --output=vina_cpu_%A_%a.out
#SBATCH --error=vina_cpu_%A_%a.err
#SBATCH --array=1-15
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G

# File: scripts/vina_cpu_array_with_progress.sh

set -euo pipefail
set -x

# --- environment ---
module purge || true
module load AutoDock-Vina/1.2.3-foss-2021b

command -v vina >/dev/null 2>&1 || { echo "ERROR: 'vina' not found on PATH."; exit 2; }

# --- config ---
VINA_SEED=42
EXHAUSTIVENESS="${EXHAUSTIVENESS:-8}"
CPU_THREADS="${CPU_THREADS:-${SLURM_CPUS_PER_TASK:-2}}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"   # print a progress line every N ligands queued

MASTER_CSV="${MASTER_CSV:-$SLURM_SUBMIT_DIR/LIT_PCBA/vina_boxes.csv}"
CSV_DIR="$(dirname "$MASTER_CSV")"
cd "$CSV_DIR" || exit
MASTER_CSV="$(basename "$MASTER_CSV")"

# Derived concurrency: total threads per task divided by per-vina threads
LIG_CHUNK_SIZE="${LIG_CHUNK_SIZE:-1000}"  # unused now but kept for compatibility
MAX_PARALLEL=$(( SLURM_CPUS_PER_TASK / CPU_THREADS ))
[[ $MAX_PARALLEL -ge 1 ]] || MAX_PARALLEL=1

LOCAL_ROOT="${TMPDIR:-/tmp}"

# --- helpers: csv ---
csv_total_lines() { awk 'NR>1 && NF>0 {c++} END{print c+0}' "$MASTER_CSV"; }
csv_get_line()    { tail -n +2 "$MASTER_CSV" | sed -n "${1}p"; }
parse_row() {
  python3 - "$1" <<'PY'
import csv, sys
row = next(csv.reader([sys.argv[1]]))
keys = [
  "target","receptor_pdbqt","actives_glob","inactives_glob","out_dir",
  "center_x","center_y","center_z","size_x","size_y","size_z","thread",
  "rescoring_receptor_pdb","sdf_out_actives_dir","sdf_out_inactives_dir","aev_plig_csv"
]
d = dict(zip(keys, row))
for k,v in d.items():
    print(f'{k}={v!r}')
PY
}

# --- helpers: lightweight semaphore for N-way parallelism ---
_parallel_init() {
  SEM_FIFO="$(mktemp -u)"; mkfifo "$SEM_FIFO"
  exec 9<>"$SEM_FIFO"; rm -f "$SEM_FIFO"
  local n="${MAX_PARALLEL:-1}"; for _ in $(seq 1 "$n"); do echo >&9; done
}
_parallel_run() { read -u 9; { "$@"; ec=$?; echo >&9; exit $ec; } & }
_parallel_wait() { wait; exec 9>&- 9<&-; }

# --- core runner ---
run_class() {
  local CLASS="$1" GLOB="$2" OUT_DIR="$3" RECEPTOR="$4"
  local CX="$5" CY="$6" CZ="$7" SX="$8" SY="$9" SZ="${10}"

  shopt -s nullglob
  mapfile -t LIGS < <(eval "printf '%s\n' ${GLOB}")
  shopt -u nullglob
  [[ ${#LIGS[@]} -gt 0 ]] || { echo "No ${CLASS} ligands; skipping."; return; }

  mkdir -p "${OUT_DIR}/${CLASS}"

  # Stage inputs to node-local scratch (faster I/O)
  local STAGE_ROOT="${LOCAL_ROOT}/vina_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${CLASS}"
  mkdir -p "${STAGE_ROOT}/ligands"
  cp -f "$RECEPTOR" "${STAGE_ROOT}/receptor.pdbqt"
  cp -f "${LIGS[@]}" "${STAGE_ROOT}/ligands/"
  # Repoint ligand list to staged copies
  local i
  for i in "${!LIGS[@]}"; do
    LIGS[$i]="${STAGE_ROOT}/ligands/$(basename "${LIGS[$i]}")"
  done

  # Base Vina config shared by all ligands of this class
  local BASE_CFG="${STAGE_ROOT}/vina_base.cfg"
  {
    echo "receptor = ${STAGE_ROOT}/receptor.pdbqt"
    echo "center_x = ${CX}"
    echo "center_y = ${CY}"
    echo "center_z = ${CZ}"
    echo "size_x = ${SX}"
    echo "size_y = ${SY}"
    echo "size_z = ${SZ}"
    echo "exhaustiveness = ${EXHAUSTIVENESS}"
  } > "$BASE_CFG"

  local total=${#LIGS[@]}
  local queued=0 skipped=0
  echo "Submitting ${total} ${CLASS} ligands with up to ${MAX_PARALLEL} concurrent Vina jobs (cpu=${CPU_THREADS}/job)"

  _parallel_init
  local lig name out_pdbqt
  for lig in "${LIGS[@]}"; do
    name="$(basename "${lig%.pdbqt}")"
    out_pdbqt="${OUT_DIR}/${CLASS}/${name}_out.pdbqt"

    # Idempotent: skip if output already exists and is non-empty
    if [[ -s "$out_pdbqt" ]]; then
      ((skipped++)) || true
      continue
    fi

    _parallel_run bash -c '
      set -euo pipefail
      BASE_CFG="$1"; lig="$2"; out_pdbqt="$3"; cpu="$4"; seed="$5"
      mkdir -p "$(dirname "$out_pdbqt")"
      vina --config "$BASE_CFG" \
           --ligand "$lig" \
           --out "$out_pdbqt" \
           --cpu "$cpu" \
           --seed "$seed"
      echo "Wrote: $out_pdbqt"
    ' _ "$BASE_CFG" "$lig" "$out_pdbqt" "$CPU_THREADS" "$VINA_SEED"

    ((queued++)) || true
    if (( queued % PROGRESS_EVERY == 0 )); then
      echo "Progress [${CLASS}]: ${queued} / ${total} queued"
    fi
  done
  _parallel_wait

  if (( queued % PROGRESS_EVERY != 0 )); then
    echo "Progress [${CLASS}]: ${queued} / ${total} queued"
  fi

  echo "Completed [${CLASS}]: attempted=${queued}, skipped_existing=${skipped}, total_listed=${total}"

  rm -rf "$STAGE_ROOT"
}

# --- main ---
ROW="${SLURM_ARRAY_TASK_ID}"
TOTAL="$(csv_total_lines)"
(( ROW >=1 && ROW <= TOTAL )) || { echo "Row index $ROW out of range"; exit 1; }
RAW="$(csv_get_line "$ROW")"
eval "$(parse_row "$RAW")"

mkdir -p "${out_dir}/actives" "${out_dir}/inactives"

run_class actives   "$actives_glob"   "$out_dir" "$receptor_pdbqt" \
          "$center_x" "$center_y" "$center_z" "$size_x" "$size_y" "$size_z"
run_class inactives "$inactives_glob" "$out_dir" "$receptor_pdbqt" \
          "$center_x" "$center_y" "$center_z" "$size_x" "$size_y" "$size_z"

echo "Protein task $ROW complete."
