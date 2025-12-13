#!/usr/bin/env bash
# File: scripts/gpu_quick_pick.sh
# Goal: Quick, robust way to decide *which GPU* (model/constraint) you should request for Vina-GPU.
# Usage:
#   bash scripts/gpu_quick_pick.sh /path/to/vina-gpu-binary
# Notes:
#   - Works on login or compute nodes. On compute nodes, honors CUDA_VISIBLE_DEVICES.
#   - If cuobjdump is missing, falls back to strings-based heuristic.
#   - Prints suggested #SBATCH line(s) and whether your binary likely supports the local GPU.

set -Eeuo pipefail
BIN=${1:-vina-gpu}

msg(){ printf "[%s] %s\n" "$(date +%T)" "$*"; }
which_or(){ command -v "$1" >/dev/null 2>&1 || command -v "$2" >/dev/null 2>&1; }

# --- 1) What GPUs are present? ---
list_gpus(){
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader 2>/dev/null || nvidia-smi -L || true
  else
    echo "No nvidia-smi found"; return 1
  fi
}

# --- 2) Compute capability quick read (best-effort) ---
compute_caps(){
  if nvidia-smi --help-query-gpu | grep -q compute_cap; then
    nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader 2>/dev/null || true
  else
    # fallback: use nvcc or deviceQuery (if available)
    if command -v deviceQuery >/dev/null; then
      deviceQuery | awk -F: '/Device [0-9]+/{g=$0} /CUDA Capability/{print g,", compute_cap=",$2}'
    else
      echo "(compute_cap unavailable via nvidia-smi; skipping)"
    fi
  fi
}

# --- 3) What SM architectures does the binary support? ---
list_binary_sms(){
  local b="$1"
  if command -v cuobjdump >/dev/null; then
    cuobjdump --list-elf "$b" 2>/dev/null | awk '/arch =/ {print $NF}' | sort -u | tr '\n' ' '; echo
  else
    strings "$b" 2>/dev/null | grep -o 'sm_[0-9][0-9]' | sort -u | tr '\n' ' '; echo
  fi
}

sm_to_cc(){ # translate sm tag ? compute capability major.minor (approx)
  case "$1" in
    sm_70) echo 7.0;; sm_72) echo 7.2;; sm_75) echo 7.5;;
    sm_80) echo 8.0;; sm_86) echo 8.6;; sm_89) echo 8.9;;
    sm_90) echo 9.0;; sm_90a) echo 9.0;;
    *) echo "?";;
  esac
}

model_to_need(){ # gpu model ? suggested constraint + needed CC
  local m="$1"; m=${m,,}
  if [[ $m == *"h100"* ]]; then echo "constraint=h100 need_cc=9.0"; return; fi
  if [[ $m == *"a100"* ]]; then echo "constraint=a100 need_cc=8.0"; return; fi
  if [[ $m == *"v100"* ]]; then echo "constraint=v100 need_cc=7.0"; return; fi
  if [[ $m == *"t4"* ]]; then echo "constraint=t4 need_cc=7.5"; return; fi
  if [[ $m == *"rtx 30"* || $m == *"a40"* || $m == *"a5000"* ]]; then echo "constraint=ampere need_cc=8.x"; return; fi
  echo "constraint=gpu need_cc=?";
}

suggest_sbatch(){
  local model="$1"; local s; s=$(model_to_need "$model")
  local c=$(awk '{print $1}' <<<"$s" | cut -d= -f2)
  if [[ $c == h100 || $c == a100 || $c == v100 || $c == t4 ]]; then
    echo "#SBATCH --constraint=$c    # or: #SBATCH --gres=gpu:${c}:1"
  else
    echo "#SBATCH --gres=gpu:1       # (generic)"
  fi
}

main(){
  msg "GPU(s) on this node:"; list_gpus || true
  msg "Compute capability:"; compute_caps || true

  msg "Vina-GPU binary: $BIN"
  if ! command -v "$BIN" >/dev/null && [[ ! -x "$BIN" ]]; then
    msg "WARN: binary not on PATH and not executable: $BIN"
  fi

  local sms; sms=$(list_binary_sms "$BIN")
  msg "Binary supports SM tags: ${sms:-unknown}"

  # If a GPU model is known, print a suggested #SBATCH line
  if command -v nvidia-smi >/dev/null; then
    local model; model=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 2>/dev/null || true)
    if [[ -n "$model" ]]; then
      msg "Suggested scheduler request for this model ($model):"
      suggest_sbatch "$model"
      # Quick compatibility hint
      if [[ "$sms" == *sm_90* ]] && [[ "$model" == *H100* ]]; then
        msg "OK: binary likely compatible with H100 (sm_90)."
      elif [[ "$sms" != *sm_90* && "$model" == *H100* ]]; then
        msg "Heads-up: no sm_90 found; needs PTX to JIT or a rebuild for H100."
      fi
    fi
  fi

  # Bonus: show partitions/features if sinfo is available
  if command -v sinfo >/dev/null; then
    msg "Cluster partitions/features (grep gpu):"
    sinfo -o '%20P %10G %40f' | grep -i gpu || true
  fi
}

main "$@"
