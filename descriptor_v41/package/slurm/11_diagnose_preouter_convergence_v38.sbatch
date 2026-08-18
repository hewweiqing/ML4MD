#!/bin/bash -l
#SBATCH --job-name=alignn-v38-ts-diagnostic
#SBATCH --partition=gpu-a100-small
#SBATCH --account=research-ME-mse
#SBATCH --time=00:30:00
#SBATCH --array=6,14,16%3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --output=logs/v38-ts-diagnostic-%A_%a.out
#SBATCH --error=logs/v38-ts-diagnostic-%A_%a.err
set -euo pipefail
module load 2025
: "${SCRATCH:=/scratch/$USER}"
ENV_PREFIX="${ALIGNN_ENV_PREFIX:-$SCRATCH/conda_envs/alignn_matbench_is_metal_cu118_v9}"
STAGING_ROOT="${ALIGNN_STAGING_ROOT:-/home/$USER/ml4md/ALIGNN/.stage2_primary_v26_staging}"
V26_ROOT="${ALIGNN_V26_PACKAGE_ROOT:-/home/$USER/alignn_stage2_v26/delftblue_package_v26}"
V36_ROOT="${ALIGNN_V36_PACKAGE_ROOT:-/home/$USER/alignn_stage2_v36/delftblue_package_v36}"
FINAL_ROOT="${ALIGNN_WORK_ROOT:-/home/$USER/ml4md/ALIGNN/stage2_primary_v26}"
CACHE_ROOT="${ALIGNN_GRAPH_CACHE_ROOT:-$SCRATCH/alignn_matbench_is_metal_graph_cache_v26_verified}"
FOLD=$((SLURM_ARRAY_TASK_ID / 5)); SEED=$((SLURM_ARRAY_TASK_ID % 5))
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs preflight/v38_convergence
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" DGLBACKEND=pytorch OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export ALIGNN_ENV_PREFIX="$ENV_PREFIX"
source scripts/activate_cuda_runtime.sh
srun bash scripts/require_v37_external_gates.sh "$ENV_PREFIX" "$V26_ROOT" "$V36_ROOT" "$CACHE_ROOT" "$FINAL_ROOT"
srun "$ENV_PREFIX/bin/python" scripts/diagnose_v38_preouter_convergence.py \
  --cell "$STAGING_ROOT/fold_${FOLD}/seed_${SEED}" --fold "$FOLD" --seed "$SEED" \
  --output "preflight/v38_convergence/fold_${FOLD}_seed_${SEED}.json"
