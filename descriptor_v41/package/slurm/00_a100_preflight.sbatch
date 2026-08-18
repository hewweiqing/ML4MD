#!/bin/bash -l
#SBATCH --job-name=alignn-s2-preflight
#SBATCH --partition=gpu-a100
#SBATCH --account=research-ME-mse
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/preflight-%j.out
#SBATCH --error=logs/preflight-%j.err
set -euo pipefail
module load 2025
: "${SCRATCH:=/scratch/$USER}"
ENV_PREFIX="${ALIGNN_ENV_PREFIX:-$SCRATCH/conda_envs/alignn_matbench_is_metal_cu118_v9}"
DATASET="${ALIGNN_DATASET:-$SCRATCH/alignn_matbench_is_metal_data/matbench_mp_is_metal.json.gz}"
PERSISTENT_ROOT="${ALIGNN_WORK_ROOT:-/home/$USER/ml4md/ALIGNN/stage2_primary_v20}"
mkdir -p "$SLURM_SUBMIT_DIR/logs" "$SLURM_SUBMIT_DIR/preflight"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export ALIGNN_ENV_PREFIX="$ENV_PREFIX"
source scripts/activate_cuda_runtime.sh
srun "$ENV_PREFIX/bin/python" scripts/require_certification.py --kind login \
  --path "$ENV_PREFIX/.alignn_stage2_v9_login_certification.json" --expected-prefix "$ENV_PREFIX"
srun "$ENV_PREFIX/bin/python" scripts/a100_preflight.py \
  --output "preflight/A100_RESOURCE_PREFLIGHT_${SLURM_JOB_ID}.json" \
  --certification-output "preflight/A100_RUNTIME_CERTIFICATION.json" \
  --dataset "$DATASET" --writable-root "$PERSISTENT_ROOT" \
  --login-certification "$ENV_PREFIX/.alignn_stage2_v9_login_certification.json" \
  --package-manifest "$SLURM_SUBMIT_DIR/PACKAGE_MANIFEST.json"
echo "ALIGNN_ENVIRONMENT: PASS"
