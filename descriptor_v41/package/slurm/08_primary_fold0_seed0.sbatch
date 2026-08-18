#!/bin/bash -l
#SBATCH --job-name=alignn-primary-f0s0
#SBATCH --partition=gpu-a100
#SBATCH --account=research-ME-mse
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --output=logs/primary-f0s0-%j.out
#SBATCH --error=logs/primary-f0s0-%j.err
set -euo pipefail
module load 2025
: "${SCRATCH:=/scratch/$USER}"
ENV_PREFIX="${ALIGNN_ENV_PREFIX:-$SCRATCH/conda_envs/alignn_matbench_is_metal_cu118_v9}"
DATASET="${ALIGNN_DATASET:-$SCRATCH/alignn_matbench_is_metal_data/matbench_mp_is_metal.json.gz}"
FINAL_ROOT="${ALIGNN_WORK_ROOT:-/home/$USER/ml4md/ALIGNN/stage2_primary_v20}"
WORK_ROOT="${ALIGNN_STAGING_ROOT:-/home/$USER/ml4md/ALIGNN/.stage2_primary_v20_staging}"
QUARANTINE_ROOT="${ALIGNN_QUARANTINE_ROOT:-/home/$USER/ml4md/ALIGNN/quarantine_v20}"
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs "$WORK_ROOT"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" DGLBACKEND=pytorch
export ALIGNN_GRAPH_CACHE_ROOT="${ALIGNN_GRAPH_CACHE_ROOT:-$SCRATCH/alignn_matbench_is_metal_graph_cache_v20}"
export ALIGNN_ENV_PREFIX="$ENV_PREFIX"
source scripts/activate_cuda_runtime.sh
srun "$ENV_PREFIX/bin/python" scripts/require_certification.py --kind a100 --path preflight/A100_RUNTIME_CERTIFICATION.json --expected-prefix "$ENV_PREFIX"
srun bash scripts/require_profile_current.sh "$ENV_PREFIX/bin/python" "$ALIGNN_GRAPH_CACHE_ROOT"
set +e
srun "$ENV_PREFIX/bin/python" scripts/check_complete.py --final-root "$FINAL_ROOT" --fold 0 --seed 0
CHECK_RC=$?
set -e
if [ "$CHECK_RC" -eq 0 ]; then echo "ALIGNN_FOLD0_SEED0_PRIMARY: PASS"; exit 0; fi
if [ "$CHECK_RC" -eq 2 ]; then exit 2; fi
srun "$ENV_PREFIX/bin/python" scripts/train_paired.py --dataset "$DATASET" --work-root "$WORK_ROOT" --fold 0 --seed 0
srun "$ENV_PREFIX/bin/python" scripts/calibrate_and_export.py --dataset "$DATASET" --work-root "$WORK_ROOT" --fold 0 --seed 0
srun "$ENV_PREFIX/bin/python" scripts/verify_cell.py --cell-dir "$WORK_ROOT/fold_0/seed_0" --fold 0 --seed 0
srun "$ENV_PREFIX/bin/python" scripts/promote_cell.py --staging-root "$WORK_ROOT" --final-root "$FINAL_ROOT" --quarantine-root "$QUARANTINE_ROOT" --fold 0 --seed 0
srun "$ENV_PREFIX/bin/python" scripts/write_primary_gate.py --final-root "$FINAL_ROOT" --output preflight/PRIMARY_FOLD0_GATE_PASSED.json
echo "ALIGNN_FOLD0_SEED0_PRIMARY: PASS"
