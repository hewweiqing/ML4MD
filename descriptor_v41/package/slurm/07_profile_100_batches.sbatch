#!/bin/bash -l
#SBATCH --job-name=alignn-profile100
#SBATCH --partition=gpu-a100
#SBATCH --account=research-ME-mse
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --output=logs/profile100-%j.out
#SBATCH --error=logs/profile100-%j.err
set -euo pipefail
module load 2025
: "${SCRATCH:=/scratch/$USER}"
ENV_PREFIX="${ALIGNN_ENV_PREFIX:-$SCRATCH/conda_envs/alignn_matbench_is_metal_cu118_v9}"
DATASET="${ALIGNN_DATASET:-$SCRATCH/alignn_matbench_is_metal_data/matbench_mp_is_metal.json.gz}"
GATE_CACHE="${ALIGNN_GRAPH_CACHE_ROOT:-$SCRATCH/alignn_matbench_is_metal_graph_cache_v20}"
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs preflight
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" DGLBACKEND=pytorch
export ALIGNN_ENV_PREFIX="$ENV_PREFIX"
source scripts/activate_cuda_runtime.sh
srun "$ENV_PREFIX/bin/python" scripts/require_certification.py --kind a100 --path preflight/A100_RUNTIME_CERTIFICATION.json --expected-prefix "$ENV_PREFIX"
srun "$ENV_PREFIX/bin/python" scripts/profile_100_batches.py --dataset "$DATASET" --cache-root "$GATE_CACHE" \
  --output preflight/FULL_CONFIG_100_BATCH_PROFILE.json \
  --certification preflight/A100_RUNTIME_CERTIFICATION.json --package-manifest PACKAGE_MANIFEST.json \
  --execution-plan STAGE_2_EXECUTION_PLAN.json --resource-policy RESOURCE_POLICY.json \
  --primary-slurm slurm/08_primary_fold0_seed0.sbatch
