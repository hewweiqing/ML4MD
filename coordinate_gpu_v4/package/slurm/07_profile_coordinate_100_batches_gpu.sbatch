#!/bin/bash
#SBATCH --job-name=alignn-coordinate-gpu-profile
#SBATCH --account=research-ME-mse
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/gpu-coordinate-profile-%j.out
#SBATCH --error=logs/gpu-coordinate-profile-%j.err
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" DGLBACKEND=pytorch
: "${ALIGNN_ENV_PREFIX:=/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9}"
: "${ALIGNN_DATASET:=/scratch/$USER/alignn_matbench_is_metal_data/matbench_mp_is_metal.json.gz}"
: "${ALIGNN_GRAPH_CACHE_ROOT:=/scratch/$USER/alignn_matbench_is_metal_graph_cache_v26_verified}"
: "${ALIGNN_COORDINATE_CACHE_ROOT:=/scratch/$USER/alignn_random2_coordinate_cache_v29}"
: "${ALIGNN_V29_ROOT:=/home/$USER/alignn_stage2_v29/delftblue_package_v29}"
export ALIGNN_COORDINATE_SIGMA_AUTHORIZATION="${ALIGNN_COORDINATE_SIGMA_AUTHORIZATION:-$ALIGNN_V29_ROOT/preflight/VERIFIED_COORDINATE_SIGMA_AUTHORIZATION.json}"
source scripts/activate_cuda_runtime.sh
"$ALIGNN_ENV_PREFIX/bin/python" scripts/require_gpu_coordinate_gates.py --preflight preflight/GPU_COORDINATE_A100_PREFLIGHT.json
srun "$ALIGNN_ENV_PREFIX/bin/python" scripts/profile_coordinate_gpu.py \
  --dataset "$ALIGNN_DATASET" --clean-cache-root "$ALIGNN_GRAPH_CACHE_ROOT" \
  --coordinate-cache-root "$ALIGNN_COORDINATE_CACHE_ROOT" \
  --profile-work "/scratch/$USER/alignn_coordinate_gpu_profile_v1" \
  --output preflight/GPU_COORDINATE_100_BATCH_PROFILE.json
