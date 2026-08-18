#!/bin/bash
#SBATCH --job-name=alignn-stage3-profile-full-network-seed0
#SBATCH --account=research-ME-mse
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage-a-profile-full-network-seed0-%j.out
#SBATCH --error=logs/stage-a-profile-full-network-seed0-%j.err
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" DGLBACKEND=pytorch
: "${ALIGNN_ENV_PREFIX:=/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9}"
: "${ALIGNN_DATASET:?set ALIGNN_DATASET to the matbench_mp_is_metal dataset path}"
source scripts/activate_cuda_runtime.sh
"$ALIGNN_ENV_PREFIX/bin/python" scripts/require_stage_a_gates.py \
  --preflight preflight/STAGE_A_A100_PREFLIGHT.json
srun "$ALIGNN_ENV_PREFIX/bin/python" scripts/profile_full_network_seed0.py \
  --dataset "$ALIGNN_DATASET" --fold 0 --seed 0
