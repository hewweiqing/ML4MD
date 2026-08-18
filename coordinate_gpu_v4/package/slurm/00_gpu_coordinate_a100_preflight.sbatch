#!/bin/bash
#SBATCH --job-name=alignn-coordinate-gpu-preflight
#SBATCH --account=research-ME-mse
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/gpu-coordinate-preflight-%j.out
#SBATCH --error=logs/gpu-coordinate-preflight-%j.err
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" DGLBACKEND=pytorch
: "${ALIGNN_ENV_PREFIX:=/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9}"
source scripts/activate_cuda_runtime.sh
srun "$ALIGNN_ENV_PREFIX/bin/python" scripts/gpu_coordinate_preflight.py \
  --output preflight/GPU_COORDINATE_A100_PREFLIGHT.json
