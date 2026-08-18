#!/bin/bash
# Source-only loader for CUDA-linked DGL on CPU nodes. It never exposes or selects a GPU.
set -euo pipefail
: "${ALIGNN_ENV_PREFIX:=/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9}"
_nvidia_root="$ALIGNN_ENV_PREFIX/lib/python3.10/site-packages/nvidia"
_paths=("$ALIGNN_ENV_PREFIX/lib" "$_nvidia_root/cuda_runtime/lib" "$_nvidia_root/cublas/lib" \
  "$_nvidia_root/cusolver/lib" "$_nvidia_root/cusparse/lib")
_joined=""
for _path in "${_paths[@]}"; do
  [[ -d "$_path" ]] || { echo "Missing certified DGL link dependency directory: $_path" >&2; return 41 2>/dev/null || exit 41; }
  _joined="${_joined:+$_joined:}$_path"
done
export LD_LIBRARY_PATH="$_joined${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=""
unset _nvidia_root _paths _joined _path
