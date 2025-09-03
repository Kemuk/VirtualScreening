#!/bin/bash
#SBATCH --job-name=vina_array
#SBATCH --output=vina_%A_%a.out
#SBATCH --error=vina_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7

set -euo pipefail

# --- functions ---------------------------------------------------------------

# Strip waters (HOH/WAT/H2O) from PDBQT by residue name (columns 18-20, 1-based)
# Replace previous strip_waters_pdbqt with this safer version
strip_waters_pdbqt () {
  local in_pdbqt="$1"
  local out_pdbqt="$2"
  awk '
    # Keep only receptor-legal records
    /^ATOM  / || /^HETATM/ {
      # Remove water residues by residue name (cols 18-20)
      resn=substr($0,18,3);
      if (resn!="HOH" && resn!="WAT" && resn!="H2O") print;
      next
    }
    /^REMARK/ || /^TER/ || /^END$/ || /^MODEL/ || /^ENDMDL/ || /^CONECT/ { print; next }
    # Drop everything else (ROOT, BRANCH, TORSDOF, etc.)
    { next }
  ' "$in_pdbqt" > "$out_pdbqt"

  # Validate: ensure no ligand-only tags remain
  if grep -Eq '^(ROOT|BRANCH|ENDBRANCH|ENDROOT|TORSDOF)\b' "$out_pdbqt"; then
    echo "Error: ligand-only tags remain in receptor PDBQT ($out_pdbqt)" >&2
    return 1
  fi
}

# (Optional) Strip waters from PDB; same idea, if you ever run before PDBQT conversion
strip_waters_pdb () {
  local in_pdb="$1"
  local out_pdb="$2"
  awk 'BEGIN{FS="";OFS=""}
       $1$2$3$4$5$6=="ATOM  " || $1$2$3$4$5$6=="HETATM" {
         resn=substr($0,18,3);
         if (resn!="HOH" && resn!="WAT" && resn!="H2O") { print $0 }
         next
       }
       { print $0 }' "$in_pdb" > "$out_pdb"
}

# ---------------------------------------------------------------------------

# Load environment
module purge
module load AutoDock-Vina/1.2.3-foss-2021b

# Set paths
LIGAND_DIR="./ligands"
PROTEIN_DIR="./proteins"
RESULTS_DIR="./results"
LIST_FILE="ligand_list.txt"

# Create results & temp directory per task
mkdir -p "$RESULTS_DIR"
TMPDIR_TASK="${TMPDIR:-/tmp}/vina_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR_TASK"
trap 'rm -rf "$TMPDIR_TASK"' EXIT

# Get ligand name for this task
base=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST_FILE")

ligand="${LIGAND_DIR}/${base}_ligand.pdbqt"
protein="${PROTEIN_DIR}/${base}.pdbqt"
clean_protein="${TMPDIR_TASK}/${base}_clean.pdbqt"
out_file="${RESULTS_DIR}/${base}_out.pdbqt"
log_file="${RESULTS_DIR}/${base}_log.txt"

# Define docking box parameters
CENTER_X=64.193
CENTER_Y=17.197
CENTER_Z=12.376

SIZE_X=18.6
SIZE_Y=18.2
SIZE_Z=18.2

# Run Vina if inputs exist
if [[ -f "$protein" && -f "$ligand" ]]; then
    echo "[$(date)] Cleaning waters for $base ..."
    strip_waters_pdbqt "$protein" "$clean_protein"

    echo "[$(date)] Running Vina for $base on task $SLURM_ARRAY_TASK_ID"
    vina --ligand "$ligand" \
         --receptor "$clean_protein" \
         --center_x $CENTER_X --center_y $CENTER_Y --center_z $CENTER_Z \
         --size_x $SIZE_X --size_y $SIZE_Y --size_z $SIZE_Z \
         --cpu $SLURM_CPUS_PER_TASK \
         --out "$out_file" 
else
    echo "[$(date)] Missing ligand or protein for $base, skipping"
fi
