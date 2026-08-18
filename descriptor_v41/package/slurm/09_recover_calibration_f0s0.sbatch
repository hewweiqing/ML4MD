#!/bin/bash -l
#SBATCH --job-name=alignn-ts-recovery-f0s0
#SBATCH --partition=gpu-a100-small
#SBATCH --account=research-ME-mse
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --output=logs/ts-recovery-f0s0-%j.out
#SBATCH --error=logs/ts-recovery-f0s0-%j.err
set -euo pipefail
module load 2025
: "${SCRATCH:=/scratch/$USER}"
ENV_PREFIX="${ALIGNN_ENV_PREFIX:-$SCRATCH/conda_envs/alignn_matbench_is_metal_cu118_v9}"
DATASET="${ALIGNN_DATASET:-$SCRATCH/alignn_matbench_is_metal_data/matbench_mp_is_metal.json.gz}"
V26_ROOT="${ALIGNN_V26_PACKAGE_ROOT:-$HOME/alignn_stage2_v26/delftblue_package_v26}"
FINAL_ROOT="${ALIGNN_WORK_ROOT:-$HOME/ml4md/ALIGNN/stage2_primary_v26}"
WORK_ROOT="${ALIGNN_STAGING_ROOT:-$HOME/ml4md/ALIGNN/.stage2_primary_v26_staging}"
QUARANTINE_ROOT="${ALIGNN_QUARANTINE_ROOT:-$HOME/ml4md/ALIGNN/quarantine_v26}"
CELL="$WORK_ROOT/fold_0/seed_0"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs preflight "$FINAL_ROOT" "$QUARANTINE_ROOT"
export DGLBACKEND=pytorch
export ALIGNN_ENV_PREFIX="$ENV_PREFIX"
export ALIGNN_GRAPH_CACHE_ROOT="${ALIGNN_GRAPH_CACHE_ROOT:-$SCRATCH/alignn_matbench_is_metal_graph_cache_v26_verified}"
source scripts/activate_cuda_runtime.sh
PYTHONPATH="$V26_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  srun "$ENV_PREFIX/bin/python" "$V26_ROOT/scripts/require_certification.py" \
    --kind a100 --path "$V26_ROOT/preflight/A100_RUNTIME_CERTIFICATION.json" --expected-prefix "$ENV_PREFIX"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "$CELL/OUTER_TEST_ACCESS_STARTED.json" ]]; then
  srun "$ENV_PREFIX/bin/python" scripts/validate_v35_existing_export.py \
    --v26-root "$V26_ROOT" --cell "$CELL" \
    --output preflight/V35_F0S0_EXISTING_EXPORT_RECOVERY_VALIDATION.json
  echo "V35_SINGLE_USE_EXPORT_REUSED: PASS"
else
  srun "$ENV_PREFIX/bin/python" scripts/validate_v31_calibration_recovery.py \
    --v26-root "$V26_ROOT" --cell "$CELL" \
    --output preflight/V31_F0S0_CALIBRATION_RECOVERY_VALIDATION.json
  srun "$ENV_PREFIX/bin/python" scripts/calibrate_and_export.py \
    --dataset "$DATASET" --work-root "$WORK_ROOT" --fold 0 --seed 0
fi
srun "$ENV_PREFIX/bin/python" scripts/verify_cell.py --cell-dir "$CELL" --fold 0 --seed 0
srun "$ENV_PREFIX/bin/python" scripts/promote_cell.py \
  --staging-root "$WORK_ROOT" --final-root "$FINAL_ROOT" \
  --quarantine-root "$QUARANTINE_ROOT" --fold 0 --seed 0
srun "$ENV_PREFIX/bin/python" scripts/write_primary_gate.py \
  --final-root "$FINAL_ROOT" --output preflight/PRIMARY_FOLD0_GATE_PASSED.json
echo "ALIGNN_V35_F0S0_CALIBRATION_RECOVERY: PASS"
