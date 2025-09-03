#!/bin/bash
#SBATCH --job-name=mol2_to_pdbqt
#SBATCH --output=mol2_to_pdbqt%j.out
#SBATCH --error=mol2_to_pdbqt%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

set -euo pipefail
set -x

# Module name as requested
module purge || true
module load OpenBabel || true
command -v obabel >/dev/null || { echo "ERROR: obabel not found after 'module load OpenBabel'"; exit 1; }

# Work inside dataset relative to submit dir
cd "$SLURM_SUBMIT_DIR/LIT_PCBA" || { echo "ERROR: LIT_PCBA not found next to submit dir"; exit 1; }

# ---- water stripping helper (your function, exported for subshells) ----
strip_waters_pdbqt () {
  local in_pdbqt="$1"
  local out_pdbqt="$2"
  awk '
    /^ATOM  / || /^HETATM/ {
      resn=substr($0,18,3);
      if (resn!="HOH" && resn!="WAT" && resn!="H2O") print;
      next
    }
    /^REMARK/ || /^TER/ || /^END$/ || /^MODEL/ || /^ENDMDL/ || /^CONECT/ { print; next }
    { next }
  ' "$in_pdbqt" > "$out_pdbqt"

  if grep -Eq '^(ROOT|BRANCH|ENDBRANCH|ENDROOT|TORSDOF)\b' "$out_pdbqt"; then
    echo "Error: ligand-only tags remain in receptor PDBQT ($out_pdbqt)" >&2
    return 1
  fi
}
export -f strip_waters_pdbqt

# ---- per-file conversion (exported so xargs can call it) ----
export OMP_NUM_THREADS=1
convert_one() {
  in="$1"
  out="${in%.mol2}.pdbqt"
  tmp="${out%.pdbqt}.tmp.pdbqt"

  # Convert MOL2 -> PDBQT
  obabel -i mol2 "$in" -o pdbqt -O "$tmp"

  # Strip waters -> final output, then clean up temp
  strip_waters_pdbqt "$tmp" "$out"
  rm -f "$tmp"
  echo "Wrote: $out"
}
export -f convert_one

# ---- find all mol2 and process in parallel ----
find . -type f -name '*.mol2' -print0 \
| xargs -0 -I{} -P "${SLURM_CPUS_PER_TASK:-1}" bash -c 'convert_one "$@"' _ {}

echo "All conversions finished (waters removed)."
