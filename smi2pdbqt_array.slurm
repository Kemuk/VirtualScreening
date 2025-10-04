#!/bin/bash -l
#SBATCH --job-name=smi2pdbqt
#SBATCH --output=log/smi2pdbqt_%A_%a.out
#SBATCH --error=log/smi2pdbqt_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --time=02:00:00

set -euo pipefail
set -x
# --- CONDA ENVIRONMENT SETUP ---
# Purge system modules to ensure a clean slate
module purge

# Initialize Conda for this script
# This makes the 'conda' command available
source $(conda info --base)/etc/profile.d/conda.sh

# Activate your specific environment from your $DATA directory
conda activate "$DATA/vscreen_env"

# --- VERIFICATION ---
# Verify that the tools from your environment are now in the PATH
command -v parallel >/dev/null 2>&1 || { echo "ERROR: parallel not found in Conda env"; exit 1; }
command -v obabel >/dev/null 2>&1 || { echo "ERROR: obabel not found in Conda env"; exit 1; }
export OMP_NUM_THREADS=1


# --- where we start ---
cd "$SLURM_SUBMIT_DIR/LIT_PCBA" || exit 1

# --- choose the protein subdirectory for this array task ---
mapfile -t DIRS < <(find . -mindepth 1 -maxdepth 1 -type d | sort)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
[ "$IDX" -ge 0 ] && [ "$IDX" -lt "${#DIRS[@]}" ] || { echo "Array index out of range"; exit 2; }
PROT_DIR="${DIRS[$IDX]}"
echo "Task $SLURM_ARRAY_TASK_ID processing: $PROT_DIR"

# --- use fast node-local scratch (your implementation is excellent) ---
TMPBASE="${TMPDIR:-$SLURM_SUBMIT_DIR/_tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
WORKROOT="$TMPBASE/${PROT_DIR##./}"
FINAL_ROOT="$PROT_DIR/pdbqt"
mkdir -p "$WORKROOT/pdbqt/actives" "$WORKROOT/pdbqt/inactives"
mkdir -p "$FINAL_ROOT/actives" "$FINAL_ROOT/inactives"

# --- on exit, copy results back (your trap is excellent) ---
trap '
    echo "Staging results back to shared storage..."
    rsync -a "$WORKROOT/pdbqt/" "$FINAL_ROOT/"
    if [[ "$TMPBASE" == "$SLURM_SUBMIT_DIR/_tmp_"* ]]; then
      rm -rf "$TMPBASE"
    fi
' EXIT

# --- core converter: reads SMILES + ID and writes <ID>.pdbqt ---
convert_smi_to_named_pdbqt() {
    local smi_file="$1"
    local out_dir="$2"
    local tag="$3"

    [ -s "$smi_file" ] || { echo "NOTE: Missing or empty $tag.smi in $PROT_DIR — skipping"; return 0; }
    echo "Converting $smi_file -> $out_dir"

    # Your awk pre-processing is perfect for creating the input for parallel
    # We will pipe its output directly into the parallel command.
    awk -v tag="$tag" '
        /^[ \t]*$/ || /^#/ { next }
        {
            sm=$1; $1=""; sub(/^[ \t]+/,""); id=$0
            if (id=="") id=sprintf("%s%d", tag, NR)
            # Sanitize the ID to create a safe filename
            gsub(/[^A-Za-z0-9._-]/, "_", id);
            gsub(/^[-.]+|_+/, "_", id);
            if (id=="") id="X"
            print sm "\t" id
        }' "$smi_file" | \
    parallel \
        --colsep '\t' \
        --eta \
        -j "${SLURM_CPUS_PER_TASK}" \
        '
        smiles="{1}"; clean_id="{2}"; out_dir="'"$out_dir"'"
        out_path="${out_dir}/${clean_id}.pdbqt"
        # Skip if already exists
        [ -e "$out_path" ] && exit 0
        obabel -:"$smiles" -opdbqt --gen3d -p 7.4 --partialcharge gasteiger -O "$out_path"
        '
}

# --- run both sets into node-local scratch ---
convert_smi_to_named_pdbqt "$PROT_DIR/actives.smi"   "$WORKROOT/pdbqt/actives"   "actives"
convert_smi_to_named_pdbqt "$PROT_DIR/inactives.smi" "$WORKROOT/pdbqt/inactives" "inactives"

echo "Done: $PROT_DIR"
