#!/bin/bash
#SBATCH --job-name=alignn-stage3-stage-a-preflight
#SBATCH --account=research-ME-mse
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/stage-a-preflight-%j.out
#SBATCH --error=logs/stage-a-preflight-%j.err
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" DGLBACKEND=pytorch
: "${ALIGNN_ENV_PREFIX:=/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9}"
source scripts/activate_cuda_runtime.sh
srun "$ALIGNN_ENV_PREFIX/bin/python" scripts/stage_a_preflight.py
