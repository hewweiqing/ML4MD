#!/bin/bash
#SBATCH --job-name=alignn-coordinate-gpu-smoke
#SBATCH --account=research-ME-mse
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/gpu-coordinate-smoke-%j.out
#SBATCH --error=logs/gpu-coordinate-smoke-%j.err
set -euo pipefail
export ALIGNN_NON_PRIMARY_SMOKE=1
export ALIGNN_GPU_COORDINATE_REQUIRE_PROFILE=0
export ALIGNN_GPU_COORDINATE_WORK_ROOT="${ALIGNN_GPU_COORDINATE_SMOKE_ROOT:-/home/$USER/ml4md/ALIGNN/non_primary_gpu_coordinate_smoke_v1}"
bash scripts/run_gpu_coordinate_cell.sh 0 0
echo "ALIGNN_GPU_COORDINATE_NON_PRIMARY_SMOKE: PASS"
