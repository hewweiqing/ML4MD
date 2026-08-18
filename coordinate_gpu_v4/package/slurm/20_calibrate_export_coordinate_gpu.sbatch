#!/bin/bash
#SBATCH --job-name=alignn-coordinate-gpu-cal
#SBATCH --account=research-ME-mse
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --array=0-24%5
#SBATCH --output=logs/gpu-coordinate-cal-%A_%a.out
#SBATCH --error=logs/gpu-coordinate-cal-%A_%a.err
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" DGLBACKEND=pytorch
: "${ALIGNN_ENV_PREFIX:=/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9}"
: "${ALIGNN_DATASET:=/scratch/$USER/alignn_matbench_is_metal_data/matbench_mp_is_metal.json.gz}"
: "${ALIGNN_GRAPH_CACHE_ROOT:=/scratch/$USER/alignn_matbench_is_metal_graph_cache_v26_verified}"
: "${ALIGNN_GPU_COORDINATE_WORK_ROOT:=/home/$USER/ml4md/ALIGNN/gpu_coordinate_primary_v4}"
: "${ALIGNN_V29_ROOT:=/home/$USER/alignn_stage2_v29/delftblue_package_v29}"
export ALIGNN_COORDINATE_SIGMA_AUTHORIZATION="${ALIGNN_COORDINATE_SIGMA_AUTHORIZATION:-$ALIGNN_V29_ROOT/preflight/VERIFIED_COORDINATE_SIGMA_AUTHORIZATION.json}"
source scripts/activate_cuda_runtime.sh
"$ALIGNN_ENV_PREFIX/bin/python" scripts/require_gpu_coordinate_gates.py --preflight preflight/GPU_COORDINATE_A100_PREFLIGHT.json --profile preflight/GPU_COORDINATE_100_BATCH_PROFILE.json --approval preflight/GPU_COORDINATE_PROFILE_APPROVAL.json --primary-gate preflight/GPU_COORDINATE_PRIMARY_FOLD0_GATE.json
fold=$((SLURM_ARRAY_TASK_ID / 5)); seed=$((SLURM_ARRAY_TASK_ID % 5))
srun "$ALIGNN_ENV_PREFIX/bin/python" scripts/calibrate_export_coordinate_gpu.py --dataset "$ALIGNN_DATASET" --work-root "$ALIGNN_GPU_COORDINATE_WORK_ROOT" --fold "$fold" --seed "$seed"
