#!/bin/bash
#SBATCH --job-name=smi2pdbqt_array
#SBATCH --output=smi2pdbqt_array_%A_%a.out
#SBATCH --error=smi2pdbqt_array_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --array=1-15
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=reub0582@ox.ac.uk

set -euo pipefail
set -x

# --- environment ---
module purge || true
module load OpenBabel || true
command -v obabel >/dev/null 2>&1 || { echo "ERROR: obabel not found after module load"; exit 1; }
export OMP_NUM_THREADS=1  # each obabel conversion is single-threaded

# --- where we start ---
cd "$SLURM_SUBMIT_DIR/LIT_PCBA" || { echo "ERROR: LIT_PCBA not found next to submit dir"; exit 1; }

# --- choose the protein subdirectory for this array task ---
mapfile -t DIRS < <(find . -mindepth 1 -maxdepth 1 -type d | sort)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
[ "$IDX" -ge 0 ] && [ "$IDX" -lt "${#DIRS[@]}" ] || { echo "Array index out of range"; exit 2; }
PROT_DIR="${DIRS[$IDX]}"
echo "Task $SLURM_ARRAY_TASK_ID processing: $PROT_DIR"

# --- use fast node-local scratch for heavy I/O, then copy back at the end ---
#     ARC jobs normally have $TMPDIR set; fall back to a dir under the submit dir if not.
TMPBASE="${TMPDIR:-$SLURM_SUBMIT_DIR/_tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
WORKROOT="$TMPBASE/${PROT_DIR##./}"

# destination on shared filesystem (NOTE: directory named 'pdbqt', not 'pbdqt')
FINAL_ROOT="$PROT_DIR/pdbqt"

mkdir -p "$WORKROOT/pdbqt/actives" "$WORKROOT/pdbqt/inactives"
mkdir -p "$FINAL_ROOT/actives" "$FINAL_ROOT/inactives"

# --- on exit, copy results back and clean up local scratch ---
finalcopy() {
  echo "Staging results back to shared storage..."
  rsync -a "$WORKROOT/pdbqt/" "$FINAL_ROOT/"
}
cleanup() {
  local ec=$?
  set +e
  finalcopy
  # only remove if we created a private scratch
  if [[ "$TMPBASE" == "$SLURM_SUBMIT_DIR/_tmp_"* ]]; then
    rm -rf "$TMPBASE"
  fi
  exit $ec
}
trap cleanup EXIT

# --- simple N-way parallelism using a semaphore (N = $SLURM_CPUS_PER_TASK) ---
_parallel_init() {
  SEM_FIFO="$(mktemp -u)"; mkfifo "$SEM_FIFO"
  exec 9<>"$SEM_FIFO"; rm -f "$SEM_FIFO"
  local n="${SLURM_CPUS_PER_TASK:-1}"; for _ in $(seq 1 "$n"); do echo >&9; done
}
_parallel_run() { read -u 9; { "$@"; ec=$?; echo >&9; exit $ec; } & }
_parallel_wait() { wait; exec 9>&- 9<&-; }

# --- core converter: reads SMILES + ID and writes <ID>.pdbqt ---
convert_smi_to_named_pdbqt() {
  local smi_file="$1"    # e.g., ./ABL1/actives.smi
  local out_dir="$2"     # e.g., $WORKROOT/pdbqt/actives
  local tag="$3"         # "actives" | "inactives"

  if [ ! -s "$smi_file" ]; then
    echo "NOTE: Missing or empty $tag.smi in $PROT_DIR — skipping $tag"
    return 0
  fi

  echo "Converting $smi_file -> $out_dir (filenames from 2nd column)"

  # Build a tab-separated list: SMILES<TAB>ID  (ID = the remainder of the line)
  local pairs
  pairs="$(mktemp)"
  awk -v tag="$tag" '
    BEGIN { n=0 }
    /^[ \t]*$/ || /^#/ { next }
    {
      n++
      sm=$1; $1=""; sub(/^[ \t]+/,""); id=$0
      if (id=="") id=sprintf("%s%d", tag, n)
      print sm "\t" id
    }' "$smi_file" > "$pairs"

  # sanitize IDs into safe filenames
  sanitize() { printf "%s" "$1" | tr -cd "A-Za-z0-9._-" | sed -E "s/^[-.]+//; s/_+/_/g; s/^$/X/"; }

  # mildly informative progress counter (optional)
  total=$(wc -l < "$pairs"); donecnt=0

  _parallel_init
  while IFS=$'\t' read -r smiles raw_id || [ -n "${smiles:-}" ]; do
    [ -n "${smiles:-}" ] || continue
    clean_id="$(sanitize "$raw_id")"

    _parallel_run bash -c '
      set -euo pipefail
      smiles="$1"; out_dir="$2"; base="$3"
      out="$out_dir/$base.pdbqt"
      # idempotent: skip if already present (useful on re-runs)
      [ -e "$out" ] && exit 0
      obabel -:"$smiles" -opdbqt --gen3d -p 7.4 --partialcharge gasteiger -O "$out"
      echo "Wrote: $out"
    ' _ "$smiles" "$out_dir" "$clean_id"

    # lightweight progress every ~100 ligands
    donecnt=$((donecnt+1))
    if (( donecnt % 100 == 0 )); then
      echo "Progress [$tag]: $donecnt / $total"
    fi
  done < "$pairs"

  _parallel_wait
  rm -f "$pairs"
}

# --- run both sets into node-local scratch, then trap will copy back ---
convert_smi_to_named_pdbqt "$PROT_DIR/actives.smi"   "$WORKROOT/pdbqt/actives"   "actives"
convert_smi_to_named_pdbqt "$PROT_DIR/inactives.smi" "$WORKROOT/pdbqt/inactives" "inactives"

echo "Done: $PROT_DIR"
