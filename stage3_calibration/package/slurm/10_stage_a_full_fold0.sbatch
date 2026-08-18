#!/bin/bash
#SBATCH --job-name=alignn-stage3-stage-a-full-fold0
#SBATCH --account=research-ME-mse
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage-a-full-fold0-%j.out
#SBATCH --error=logs/stage-a-full-fold0-%j.err
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" DGLBACKEND=pytorch
: "${ALIGNN_ENV_PREFIX:=/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9}"
: "${ALIGNN_DATASET:?set ALIGNN_DATASET to the matbench_mp_is_metal dataset path}"
: "${ALIGNN_STAGE_A_WORK_ROOT:?set ALIGNN_STAGE_A_WORK_ROOT to the Stage A output directory}"
: "${ALIGNN_COORDINATE_SIGMA_AUTHORIZATION:?set ALIGNN_COORDINATE_SIGMA_AUTHORIZATION to the verified authorization artifact (same value coordinate_gpu_v4 uses; see GPU_COORDINATE_V29_AUTHORIZATION_BINDING_SOURCE.md)}"
export STAGE3_COORDINATE_SIGMA_AUTHORIZATION="$ALIGNN_COORDINATE_SIGMA_AUTHORIZATION"
source scripts/activate_cuda_runtime.sh
# Requires both the A100 preflight AND an explicit human-reviewer approval
# of the single-seed full-network timing profile (05_profile_full_network_seed0.sbatch)
# before this 20-seed run is unlocked.
"$ALIGNN_ENV_PREFIX/bin/python" scripts/require_stage_a_gates.py \
  --preflight preflight/STAGE_A_A100_PREFLIGHT.json \
  --profile preflight/STAGE_A_FULL_NETWORK_SEED0_PROFILE.json \
  --approval preflight/STAGE_A_FULL_NETWORK_SEED0_PROFILE_APPROVAL.json
srun "$ALIGNN_ENV_PREFIX/bin/python" scripts/run_stage_a_diagnostic.py \
  --dataset "$ALIGNN_DATASET" --fold 0 --work-dir "$ALIGNN_STAGE_A_WORK_ROOT"
