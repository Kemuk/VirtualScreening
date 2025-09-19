#!/bin/bash
# FILE: generate_sdfs_array.sh
#SBATCH --job-name=gen_sdfs
#SBATCH --output=logs/gen_sdfs_%A_%a.out
#SBATCH --error=logs/gen_sdfs_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-1000
#SBATCH --mail-type=NONE

set -euo pipefail

module purge || true
module load Anaconda3
module load parallel
source activate /data/stat-cadd/reub0582/vscreen_env

OBABEL=/data/stat-cadd/reub0582/vscreen_env/bin/obabel
MAIL_TO="Reub0582@ox.ac.uk"

cd "$SLURM_SUBMIT_DIR/LIT_PCBA" || { echo "ERROR: LIT_PCBA not found"; exit 1; }
mkdir -p .sentinels/gen_sdfs

send_email() {
  local subject="$1"; shift
  local body="$*"
  if command -v mail >/dev/null 2>&1; then
    echo -e "$body" | mail -s "$subject" "$MAIL_TO" || true
  elif command -v sendmail >/dev/null 2>&1; then
    {
      echo "To: $MAIL_TO"
      echo "Subject: $subject"
      echo
      echo -e "$body"
    } | sendmail -t || true
  else
    echo "WARN: mail/sendmail not found; could not send email." >&2
  fi
}

PY_HELPER=$(mktemp /tmp/write_sdf_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_XXXX.py)
cat > "$PY_HELPER" << 'PYEOF'
import sys, os
from rdkit import Chem
try:
    from meeko import PDBQTMolecule, RDKitMolCreate
except Exception:
    PDBQTMolecule = None

def rdkit_from_pdbqt(path):
    if PDBQTMolecule is None: return None
    try:
        pq = PDBQTMolecule.from_file(path, skip_typing=True)
        mols = [m for m in RDKitMolCreate.from_pdbqt_mol(pq) if m is not None]
        return mols[0] if mols else None
    except Exception:
        return None

def write_sdf(pdbqt_in, sdf_out):
    mol = rdkit_from_pdbqt(pdbqt_in)
    if mol is None: return 2
    os.makedirs(os.path.dirname(sdf_out), exist_ok=True)
    try:
        Chem.MolToMolFile(mol, sdf_out)
        return 0
    except Exception:
        return 3

if __name__ == "__main__":
    sys.exit(write_sdf(sys.argv[1], sys.argv[2]))
PYEOF

emit_all() {
  for pdir in */; do
    [ -d "$pdir" ] || continue
    pdir="${pdir%/}"
    for mode in actives inactives; do
      # flat
      find "$pdir" -type f -path "*/$mode/*.pdbqt" -print0 2>/dev/null \
      | xargs -0 -I{} bash -c '
          lig="{}"; lig_id=$(basename "$lig" .pdbqt); d="'"$pdir"'/docked_vina/'"$mode"'/${lig_id}.pdbqt";
          pref="$d"; [ -f "$pref" ] || pref="$lig";
          printf "%s\t%s\t%s\t%s\t%s\n" "'"$pdir"'" "'"$mode"'" "$lig" "$lig_id" "$pref"
        '
      # nested
      find "$pdir" -type f \( -path "*/ligands/$mode/*.pdbqt" -o -path "*/input/$mode/*.pdbqt" \) -print0 2>/dev/null \
      | xargs -0 -I{} bash -c '
          lig="{}"; lig_id=$(basename "$lig" .pdbqt); d="'"$pdir"'/docked_vina/'"$mode"'/${lig_id}.pdbqt";
          pref="$d"; [ -f "$pref" ] || pref="$lig";
          printf "%s\t%s\t%s\t%s\t%s\n" "'"$pdir"'" "'"$mode"'" "$lig" "$lig_id" "$pref"
        '
    done
  done
}

COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"
ID="${SLURM_ARRAY_TASK_ID:-0}"

build_one() {
  local protein_dir="$1" mode="$2" input_pdbqt="$3" lig_id="$4" pref_pdbqt="$5"
  local sdf_dir="$protein_dir/docked_sdf/$mode"
  local lig_sdf="$sdf_dir/${lig_id}.sdf"
  [ -f "$lig_sdf" ] && return 0
  if ! python "$PY_HELPER" "$pref_pdbqt" "$lig_sdf" >/dev/null 2>&1; then
    tmp_pdb=$(mktemp /tmp/lig_${lig_id}_XXXX.pdb)
    if $OBABEL -ipdbqt "$pref_pdbqt" -opdb -O "$tmp_pdb" >/dev/null 2>&1; then
      $OBABEL -ipdb "$tmp_pdb" -osdf -O "$lig_sdf" >/dev/null 2>&1 || true
    fi
    rm -f "$tmp_pdb"
  fi
  [ -f "$lig_sdf" ] || echo "WARN: failed SDF $protein_dir/$mode/$lig_id" >&2
}
export -f build_one
export OBABEL PY_HELPER

emit_all \
| awk -v c="$COUNT" -v i="$ID" '((NR-1)%c)==i' \
| parallel -j "$SLURM_CPUS_PER_TASK" --colsep '\t' build_one {1} {2} {3} {4} {5}

rm -f "$PY_HELPER"

# Sentinel + single email
touch ".sentinels/gen_sdfs/task_${ID}.done"
{
  exec 9>".sentinels/gen_sdfs/.lock"
  flock -n 9 || exit 0
  done_count=$(ls -1 .sentinels/gen_sdfs/task_*.done 2>/dev/null | wc -l | tr -d ' ')
  if [ "$done_count" -ge "$COUNT" ]; then
    send_email "[SLURM] SDF generation finished (job ${SLURM_ARRAY_JOB_ID})" \
      "All ${COUNT} tasks completed on $(hostname).\nRoot: $PWD\nTime: $(date -Is)"
  fi
} || true

echo "Task $ID/$COUNT: SDF generation complete."
