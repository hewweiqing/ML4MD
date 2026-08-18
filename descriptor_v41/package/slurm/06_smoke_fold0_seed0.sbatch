#!/bin/bash -l
#SBATCH --job-name=alignn-smoke-f0s0
#SBATCH --partition=gpu-a100
#SBATCH --account=research-ME-mse
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err
set -euo pipefail
module load 2025
: "${SCRATCH:=/scratch/$USER}"
ENV_PREFIX="${ALIGNN_ENV_PREFIX:-$SCRATCH/conda_envs/alignn_matbench_is_metal_cu118_v9}"
DATASET="${ALIGNN_DATASET:-$SCRATCH/alignn_matbench_is_metal_data/matbench_mp_is_metal.json.gz}"
WORK_ROOT="${ALIGNN_SMOKE_ROOT:-/home/$USER/ml4md/ALIGNN/non_primary_smoke_v20}"
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs "$WORK_ROOT"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" DGLBACKEND=pytorch ALIGNN_NON_PRIMARY_SMOKE=1
export ALIGNN_RANDOM2_SYNTHETIC_COUNT=128 ALIGNN_RANDOM2_WARMUP_STEPS=4
export ALIGNN_GRAPH_CACHE_ROOT="${ALIGNN_GRAPH_CACHE_ROOT:-$SCRATCH/alignn_matbench_is_metal_graph_cache_v20}"
export ALIGNN_ENV_PREFIX="$ENV_PREFIX"
source scripts/activate_cuda_runtime.sh
srun "$ENV_PREFIX/bin/python" scripts/require_certification.py --kind a100 --path preflight/A100_RUNTIME_CERTIFICATION.json --expected-prefix "$ENV_PREFIX"
srun "$ENV_PREFIX/bin/python" scripts/verify_execution_contracts.py --device cuda
srun "$ENV_PREFIX/bin/python" scripts/train_paired.py --dataset "$DATASET" --work-root "$WORK_ROOT" --fold 0 --seed 0
echo "ALIGNN_NON_PRIMARY_SMOKE: PASS"
