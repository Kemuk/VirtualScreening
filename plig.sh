#!/usr/bin/env bash
set -euo pipefail

# --- Config (optional CUDA allocator + thread limits) ---
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:256'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# --- Paths (we are in the parent dir of LIT_PCBA/ and AEV-PLIG/) ---
SRC_CSV="LIT_PCBA/dataset.csv"
CHUNK_DIR="LIT_PCBA/plig_chunks"

# --- Make chunk folder ---
mkdir -p "$CHUNK_DIR"

# --- Read header and split data rows into ~1000-line chunks ---
read -r header < "$SRC_CSV"
hdr="$(mktemp)"; printf '%s\n' "$header" > "$hdr"

# Create chunk files (data rows only for now)
# e.g. LIT_PCBA/plig chunks/plig_chunk_0000.csv, plig_chunk_0001.csv, ...
tail -n +2 "$SRC_CSV" \
| split -d -l 1000 -a 4 --additional-suffix=.csv - "${CHUNK_DIR}/plig_chunk_"

# Prepend header to each chunk (multi-core if GNU parallel is available)
if command -v parallel >/dev/null 2>&1; then
  find "$CHUNK_DIR" -maxdepth 1 -name 'plig_chunk_*.csv' -print0 \
  | parallel -0 -j 8 --halt soon,fail=1 '
      f={}; tmp="${f%.csv}.tmp"
      cat "'"$hdr"'" "$f" > "$tmp" && mv "$tmp" "$f"
    '
else
  # Fallback: use xargs with 8 workers
  find "$CHUNK_DIR" -maxdepth 1 -name 'plig_chunk_*.csv' -print0 \
  | xargs -0 -n1 -P 8 -I{} bash -lc '
      f="$1"; tmp="${f%.csv}.tmp"
      cat "'"$hdr"'" "$f" > "$tmp" && mv "$tmp" "$f"
    ' _ {}
fi
rm -f "$hdr"

# --- (Optional) sanity checks ---
echo "Original + chunks line counts:"
wc -l "$SRC_CSV" "${CHUNK_DIR}"/plig_chunk_*.csv | tail
echo "Total chunks: $(ls -1 "${CHUNK_DIR}"/plig_chunk_*.csv | wc -l)"

# --- Run AEV-PLIG on each chunk sequentially ---
# We cd into AEV-PLIG so its relative paths behave as expected; we pass the chunk path relative to it.
for csv in "${CHUNK_DIR}"/plig_chunk_*.csv; do
  base="$(basename "${csv%.csv}")"        # e.g. "plig_chunk_0000"
  data_name="LIT_PCBA_${base}"            # omit ".csv"
  echo "=== Running ${data_name} ==="

  (
    cd AEV-PLIG
    python process_and_predict.py \
      --dataset_csv="../${csv}" \
      --data_name="${data_name}" \
      --trained_model_name="model_GATv2Net_ligsim90_fep_benchmark" \
      --skip_validation \
      --num_workers=1 \
      --device auto
  )
done

echo "All chunks processed."
