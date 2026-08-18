#!/bin/bash
#SBATCH --job-name=alignn-coordinate-gpu-grid
#SBATCH --account=research-ME-mse
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --array=1-24%5
#SBATCH --output=logs/gpu-coordinate-train-%A_%a.out
#SBATCH --error=logs/gpu-coordinate-train-%A_%a.err
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
: "${ALIGNN_ENV_PREFIX:=/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9}"
"$ALIGNN_ENV_PREFIX/bin/python" scripts/require_gpu_coordinate_gates.py \
  --preflight preflight/GPU_COORDINATE_A100_PREFLIGHT.json \
  --profile preflight/GPU_COORDINATE_100_BATCH_PROFILE.json \
  --approval preflight/GPU_COORDINATE_PROFILE_APPROVAL.json \
  --primary-gate preflight/GPU_COORDINATE_PRIMARY_FOLD0_GATE.json
fold=$((SLURM_ARRAY_TASK_ID / 5)); seed=$((SLURM_ARRAY_TASK_ID % 5))
bash scripts/run_gpu_coordinate_cell.sh "$fold" "$seed"
