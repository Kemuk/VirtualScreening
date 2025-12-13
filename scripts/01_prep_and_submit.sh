#!/usr/bin/env bash
#SBATCH --job-name=Vina_Prep_Worker
#SBATCH --output=log/prep_worker_%A_%a.out
#SBATCH --error=log/prep_worker_%A_%a.err
#SBATCH --time=02:00:00 # Time per protein
#SBATCH --mem=8GB       # Memory per protein
#SBATCH --partition=short

set -euo pipefail
IFS=$'\n\t'

# --- Configuration ---
LIGANDS_PER_CHUNK="${LIGANDS_PER_CHUNK:-10000}"
MASTER_CSV="${SLURM_SUBMIT_DIR}/LIT_PCBA/vina_boxes.csv"
PREP_DIR="${SLURM_SUBMIT_DIR}/prepared_data"

# Get the specific line from the CSV for this array task
ROW_DATA=$(tail -n +2 "$MASTER_CSV" | sed -n "${SLURM_ARRAY_TASK_ID}p")

# Parse the data for this specific protein
IFS=, read -r target receptor_pdbqt actives_glob inactives_glob out_dir center_x center_y center_z size_x size_y size_z thread etc <<< "$ROW_DATA"

echo "==> Preparing target [${SLURM_ARRAY_TASK_ID}]: $target"

receptor_pdbqt="LIT_PCBA/${receptor_pdbqt}"
actives_glob="LIT_PCBA/${actives_glob}"
inactives_glob="LIT_PCBA/${inactives_glob}"
out_dir="LIT_PCBA/${out_dir}"
mkdir -p "$out_dir"

all_ligands_glob="${actives_glob} ${inactives_glob}"
target_prep_dir="${PREP_DIR}/${target}"
chunks_dir="${target_prep_dir}/chunks"
# Create a unique manifest for this protein to avoid race conditions
partial_manifest_file="${target_prep_dir}/manifest_${target}.txt"
mkdir -p "$chunks_dir"
rm -f "$partial_manifest_file"
touch "$partial_manifest_file"

master_ligand_list="${target_prep_dir}/all_ligands.txt"
bash -O nullglob -c 'printf "%s\n" '"$all_ligands_glob" > "$master_ligand_list"

total_ligands=$(wc -l < "$master_ligand_list")
if (( total_ligands == 0 )); then
    echo "  -> WARNING: Found 0 ligands for target ${target}. Skipping."
    exit 0
fi
echo "  -> Found ${total_ligands} total ligands."

echo "  -> Splitting into chunks of ${LIGANDS_PER_CHUNK}..."
split -l "$LIGANDS_PER_CHUNK" --numeric-suffixes=1 --additional-suffix=.txt "$master_ligand_list" "${chunks_dir}/chunk_"

ligands_tar_gz="${target_prep_dir}/ligands.tar.gz"
echo "  -> Creating ligand archive..."
tar --transform 's|.*/|ligands/|' -czf "$ligands_tar_gz" --files-from "$master_ligand_list"

# This value is a placeholder; the real thread count is set when submitting the docking job
VINA_THREADS_PLACEHOLDER="2048"
for chunk_file in "${chunks_dir}"/chunk_*.txt; do
    chunk_id=$(basename "$chunk_file" .txt | cut -d'_' -f2)
    echo "${target},${chunk_id},${receptor_pdbqt},${ligands_tar_gz},${center_x},${center_y},${center_z},${size_x},${size_y},${size_z},${out_dir},${VINA_THREADS_PLACEHOLDER}" >> "$partial_manifest_file"
done

echo "-> Preparation for $target complete."