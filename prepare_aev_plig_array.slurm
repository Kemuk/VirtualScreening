#!/bin/bash
#SBATCH --job-name=prepare_aev_plig_array
#SBATCH --output=prepare_aev_plig_array_%A_%a.out
#SBATCH --error=prepare_aev_plig_array_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-14   # set at submit: 0-$(($(ls LIT_PCBA/chunks/chunk_*.txt|wc -l)-1))

set -euo pipefail

# --- Env ---
module purge || true
module load Anaconda3
module load parallel
source activate /data/stat-cadd/reub0582/vscreen_env

OBABEL=/data/stat-cadd/reub0582/vscreen_env/bin/obabel

cd "$SLURM_SUBMIT_DIR/LIT_PCBA" || { echo "ERROR: LIT_PCBA not found"; exit 1; }

# --- Chunk selection ---
CHUNKS_DIR="chunks"
mapfile -t CHUNKS < <(ls -1 "$CHUNKS_DIR"/chunk_*.txt)
NCHUNKS=${#CHUNKS[@]}
CHUNK_FILE="${CHUNKS[$SLURM_ARRAY_TASK_ID]:-}"
[ -n "${CHUNK_FILE}" ] && [ -f "$CHUNK_FILE" ] || { echo "No chunk for index $SLURM_ARRAY_TASK_ID"; exit 1; }

OUT_CSV="${CHUNK_FILE%.txt}.csv"
echo "Ligand_ID,Protein_ID,Ligand_SDF_Path,Protein_PDB_Path,Best_Score,rmsd,MW,LogP,HBD,HBA,is_active,pred_active" > "$OUT_CSV"

# --- Minimal Python helper: write SDF from PDBQT using Meeko+RDKit ---
PY_HELPER=$(mktemp /tmp/write_sdf_${SLURM_ARRAY_TASK_ID}_XXXX.py)
cat > "$PY_HELPER" << 'PYEOF'
import sys, os
from rdkit import Chem
try:
    from meeko import PDBQTMolecule, RDKitMolCreate
except Exception:
    PDBQTMolecule = None

# Args: pdbqt_in sdf_out
if len(sys.argv) != 3:
    sys.exit(1)
pdbqt_in, sdf_out = sys.argv[1], sys.argv[2]

def rdkit_from_pdbqt(path):
    if PDBQTMolecule is None: return None
    try:
        pq = PDBQTMolecule.from_file(path, skip_typing=True)
        lst = RDKitMolCreate.from_pdbqt_mol(pq)
        mols = [m for m in lst if m is not None]
        if not mols: return None
        return mols[0]
    except Exception:
        return None

mol = rdkit_from_pdbqt(pdbqt_in)
if mol is None:
    sys.exit(2)

os.makedirs(os.path.dirname(sdf_out), exist_ok=True)
try:
    Chem.MolToMolFile(mol, sdf_out)
except Exception:
    sys.exit(3)
sys.exit(0)
PYEOF

# --- Worker (parallel) ---
process_ligand() {
  local cfg="$1"
  local input_pdbqt="$2"
  local out_csv="$3"
  local obabel="$4"
  local py_helper="$5"

  [ -f "$input_pdbqt" ] || { echo "WARN: input missing $input_pdbqt" >&2; return; }

  local protein_dir; protein_dir=$(dirname "$cfg")
  local protein_id; protein_id=$(basename "$protein_dir")
  local protein_pdb="$protein_dir/${protein_id}.pdb"

  # protein mol2 -> pdb (once)
  if [ ! -f "$protein_pdb" ]; then
    local rec_mol2; rec_mol2=$(find "$protein_dir" -maxdepth 1 -type f -name "*.mol2" | head -n1 || true)
    [ -n "$rec_mol2" ] && $obabel -imol2 "$rec_mol2" -opdb -O "$protein_pdb" >/dev/null 2>&1 || true
  fi

  local lig_id; lig_id=$(basename "$input_pdbqt" .pdbqt)
  local mode is_active
  if [[ "$input_pdbqt" == *"/actives/"* ]]; then
    mode="actives"; is_active=1
  else
    mode="inactives"; is_active=0
  fi

  local sdf_dir="$protein_dir/docked_sdf/$mode"
  local lig_sdf="$sdf_dir/${lig_id}.sdf"
  mkdir -p "$sdf_dir"

  # docked pose pdbqt (preferred for SDF)
  local dock_pdbqt="$protein_dir/docked_vina/$mode/${lig_id}.pdbqt"

  # --- make SDF (simple, robust) ---
  if [ ! -f "$lig_sdf" ]; then
    # 1) try Meeko on docked
    if [ -f "$dock_pdbqt" ]; then
      python "$py_helper" "$dock_pdbqt" "$lig_sdf" || true
    fi
    # 2) fallback: Meeko on input pdbqt
    if [ ! -f "$lig_sdf" ]; then
      python "$py_helper" "$input_pdbqt" "$lig_sdf" || true
    fi
    # 3) last resort: OpenBabel two-step (if pdbqt plugin available)
    if [ ! -f "$lig_sdf" ]; then
      tmp_pdb=$(mktemp /tmp/lig_${lig_id}_XXXX.pdb)
      if $obabel -ipdbqt "${dock_pdbqt:-$input_pdbqt}" -opdb -O "$tmp_pdb" >/dev/null 2>&1; then
        $obabel -ipdb "$tmp_pdb" -osdf -O "$lig_sdf" >/dev/null 2>&1 || true
      fi
      rm -f "$tmp_pdb"
    fi
  fi

  # --- parse best score & rmsd (mode 1) ---
  local logf="$protein_dir/docked_vina/$mode/logs/${lig_id}.log"
  local best_score="0.000" rmsd="0.000"
  if [ -f "$logf" ]; then
    read -r best_score rmsd < <(awk 'NF && $1=="1"{printf "%s %s",$2,$3; exit}' "$logf" || true)
    [ -z "$best_score" ] && best_score="0.000"
    [ -z "$rmsd" ] && rmsd="0.000"
  fi

  # --- write CSV row (props left empty for now) ---
  printf "%s,%s,%s,%s,%.3f,%.3f,,,,,%s,0\n" \
    "$lig_id" "$protein_id" "./${protein_id}/docked_sdf/${mode}/${lig_id}.sdf" "./${protein_id}/${protein_id}.pdb" \
    "$best_score" "$rmsd" "$is_active" >> "$out_csv"
}

export -f process_ligand

# --- Run ligands in parallel (within chunk) ---
parallel -j "$SLURM_CPUS_PER_TASK" --colsep '\t' process_ligand {1} {2} "$OUT_CSV" "$OBABEL" "$PY_HELPER" < "$CHUNK_FILE"

echo "Wrote chunk CSV: $OUT_CSV"

# --- Combine once (locked) ---
if [ "$SLURM_ARRAY_TASK_ID" -eq $((NCHUNKS-1)) ]; then
  LOCKFILE="chunks/.combine.lock"
  MASTER="aev_plig.csv"
  {
    flock -n 9 || exit 0
    echo "Combining chunk CSVs into $MASTER.gz ..."
    mapfile -t CHUNK_CSVS < <(ls -1 chunks/chunk_*.csv)
    [ "${#CHUNK_CSVS[@]}" -gt 0 ] || { echo "No chunk CSVs found"; exit 0; }
    head -n1 "${CHUNK_CSVS[0]}" > "$MASTER"
    tail -n+2 -q "${CHUNK_CSVS[@]}" >> "$MASTER"
    gzip -f "$MASTER"
    echo "Final: $MASTER.gz"
  } 9>"$LOCKFILE"
fi

rm -f "$PY_HELPER"
