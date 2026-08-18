#!/bin/bash -l
#SBATCH --job-name=alignn-graph-cache
#SBATCH --partition=gpu-a100
#SBATCH --account=research-ME-mse
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --output=logs/cache-%j.out
#SBATCH --error=logs/cache-%j.err
set -euo pipefail
module load 2025
: "${SCRATCH:=/scratch/$USER}"
ENV_PREFIX="${ALIGNN_ENV_PREFIX:-$SCRATCH/conda_envs/alignn_matbench_is_metal_cu118_v9}"
DATASET="${ALIGNN_DATASET:-$SCRATCH/alignn_matbench_is_metal_data/matbench_mp_is_metal.json.gz}"
CACHE_ROOT="${ALIGNN_GRAPH_CACHE_ROOT:-$SCRATCH/alignn_matbench_is_metal_graph_cache_v20}"
CACHE_QUARANTINE="${ALIGNN_CACHE_QUARANTINE_ROOT:-$SCRATCH/alignn_matbench_is_metal_graph_cache_quarantine_v20}"
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs "$CACHE_ROOT" "$CACHE_QUARANTINE"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" DGLBACKEND=pytorch
export ALIGNN_ENV_PREFIX="$ENV_PREFIX"
source scripts/activate_cuda_runtime.sh
srun "$ENV_PREFIX/bin/python" scripts/require_certification.py --kind a100 --path preflight/A100_RUNTIME_CERTIFICATION.json --expected-prefix "$ENV_PREFIX"
srun "$ENV_PREFIX/bin/python" scripts/build_graph_cache.py --dataset "$DATASET" --cache-root "$CACHE_ROOT" \
  --quarantine-root "$CACHE_QUARANTINE" --scope full_label_free --package-root "$PWD"
