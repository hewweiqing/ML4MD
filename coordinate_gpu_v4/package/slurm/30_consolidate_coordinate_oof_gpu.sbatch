#!/bin/bash
#SBATCH --job-name=alignn-coordinate-gpu-oof
#SBATCH --account=research-ME-mse
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3968M
#SBATCH --time=04:00:00
#SBATCH --output=logs/gpu-coordinate-oof-%j.out
#SBATCH --error=logs/gpu-coordinate-oof-%j.err
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
: "${ALIGNN_ENV_PREFIX:=/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9}"
: "${ALIGNN_GPU_COORDINATE_WORK_ROOT:=/home/$USER/ml4md/ALIGNN/gpu_coordinate_primary_v4}"
: "${ALIGNN_GPU_COORDINATE_OOF_ROOT:=/home/$USER/ml4md/ALIGNN/gpu_coordinate_oof_v4}"
: "${ALIGNN_V29_ROOT:=/home/$USER/alignn_stage2_v29/delftblue_package_v29}"
export ALIGNN_COORDINATE_SIGMA_AUTHORIZATION="${ALIGNN_COORDINATE_SIGMA_AUTHORIZATION:-$ALIGNN_V29_ROOT/preflight/VERIFIED_COORDINATE_SIGMA_AUTHORIZATION.json}"
srun "$ALIGNN_ENV_PREFIX/bin/python" scripts/consolidate_coordinate_oof_gpu.py --work-root "$ALIGNN_GPU_COORDINATE_WORK_ROOT" --output-dir "$ALIGNN_GPU_COORDINATE_OOF_ROOT"
