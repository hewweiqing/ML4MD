#!/bin/bash -l
#SBATCH --job-name=alignn-s2-train
#SBATCH --partition=gpu-a100
#SBATCH --account=research-ME-mse
#SBATCH --time=24:00:00
#SBATCH --array=1-24%5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --output=logs/train-%A_%a.out
#SBATCH --error=logs/train-%A_%a.err
set -euo pipefail
module load 2025
: "${SCRATCH:=/scratch/$USER}"
ENV_PREFIX="${ALIGNN_ENV_PREFIX:-$SCRATCH/conda_envs/alignn_matbench_is_metal_cu118_v9}"
DATASET="${ALIGNN_DATASET:-$SCRATCH/alignn_matbench_is_metal_data/matbench_mp_is_metal.json.gz}"
FINAL_ROOT="${ALIGNN_WORK_ROOT:-/home/$USER/ml4md/ALIGNN/stage2_primary_v26}"
WORK_ROOT="${ALIGNN_STAGING_ROOT:-/home/$USER/ml4md/ALIGNN/.stage2_primary_v26_staging}"
V26_ROOT="${ALIGNN_V26_PACKAGE_ROOT:-/home/$USER/alignn_stage2_v26/delftblue_package_v26}"
V36_ROOT="${ALIGNN_V36_PACKAGE_ROOT:-/home/$USER/alignn_stage2_v36/delftblue_package_v36}"
FOLD=$((SLURM_ARRAY_TASK_ID / 5))
SEED=$((SLURM_ARRAY_TASK_ID % 5))
mkdir -p "$SLURM_SUBMIT_DIR/logs" "$WORK_ROOT"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" DGLBACKEND=pytorch PYTHONUNBUFFERED=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export ALIGNN_GRAPH_CACHE_ROOT="${ALIGNN_GRAPH_CACHE_ROOT:-$SCRATCH/alignn_matbench_is_metal_graph_cache_v26_verified}"
export ALIGNN_ENV_PREFIX="$ENV_PREFIX"
source scripts/activate_cuda_runtime.sh
srun bash scripts/require_v37_external_gates.sh "$ENV_PREFIX" "$V26_ROOT" "$V36_ROOT" "$ALIGNN_GRAPH_CACHE_ROOT" "$FINAL_ROOT"
set +e
srun "$ENV_PREFIX/bin/python" scripts/check_complete.py --final-root "$FINAL_ROOT" --fold "$FOLD" --seed "$SEED"
CHECK_RC=$?
set -e
if [ "$CHECK_RC" -eq 0 ]; then exit 0; fi
if [ "$CHECK_RC" -eq 2 ]; then exit 2; fi
srun "$ENV_PREFIX/bin/python" scripts/train_paired.py --dataset "$DATASET" --work-root "$WORK_ROOT" --fold "$FOLD" --seed "$SEED"
