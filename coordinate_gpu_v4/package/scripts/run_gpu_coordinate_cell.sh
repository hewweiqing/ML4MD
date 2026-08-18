#!/bin/bash
set -euo pipefail

: "${ALIGNN_ENV_PREFIX:=/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9}"
: "${ALIGNN_DATASET:=/scratch/$USER/alignn_matbench_is_metal_data/matbench_mp_is_metal.json.gz}"
: "${ALIGNN_GRAPH_CACHE_ROOT:=/scratch/$USER/alignn_matbench_is_metal_graph_cache_v26_verified}"
: "${ALIGNN_COORDINATE_CACHE_ROOT:=/scratch/$USER/alignn_random2_coordinate_cache_v29}"
: "${ALIGNN_GPU_COORDINATE_WORK_ROOT:=/home/$USER/ml4md/ALIGNN/gpu_coordinate_primary_v4}"
: "${ALIGNN_V29_ROOT:=/home/$USER/alignn_stage2_v29/delftblue_package_v29}"

export ALIGNN_ENV_PREFIX ALIGNN_DATASET ALIGNN_GRAPH_CACHE_ROOT
export ALIGNN_COORDINATE_CACHE_ROOT ALIGNN_GPU_COORDINATE_WORK_ROOT ALIGNN_V29_ROOT
export ALIGNN_COORDINATE_SIGMA_AUTHORIZATION="${ALIGNN_COORDINATE_SIGMA_AUTHORIZATION:-$ALIGNN_V29_ROOT/preflight/VERIFIED_COORDINATE_SIGMA_AUTHORIZATION.json}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export DGLBACKEND=pytorch

case "$ALIGNN_GPU_COORDINATE_WORK_ROOT" in
  *cpu_coordinate*|*stage2_primary_v26*)
    echo "Refusing a CPU/descriptor experiment output root: $ALIGNN_GPU_COORDINATE_WORK_ROOT" >&2
    exit 2
    ;;
esac

source scripts/activate_cuda_runtime.sh
if [ "${ALIGNN_GPU_COORDINATE_REQUIRE_PROFILE:-1}" = "1" ]; then
  "$ALIGNN_ENV_PREFIX/bin/python" scripts/require_gpu_coordinate_gates.py \
    --preflight preflight/GPU_COORDINATE_A100_PREFLIGHT.json \
    --profile preflight/GPU_COORDINATE_100_BATCH_PROFILE.json \
    --approval preflight/GPU_COORDINATE_PROFILE_APPROVAL.json
else
  "$ALIGNN_ENV_PREFIX/bin/python" scripts/require_gpu_coordinate_gates.py \
    --preflight preflight/GPU_COORDINATE_A100_PREFLIGHT.json
fi

set +e
srun --signal=USR1@600 "$ALIGNN_ENV_PREFIX/bin/python" scripts/train_coordinate_gpu.py \
  --dataset "$ALIGNN_DATASET" \
  --work-root "$ALIGNN_GPU_COORDINATE_WORK_ROOT" \
  --fold "$1" --seed "$2" --condition paired
status=$?
set -e

if [ "$status" -eq 75 ]; then
  echo "GPU coordinate checkpoint saved; deterministic resubmission is required."
  exit 75
fi
exit "$status"
