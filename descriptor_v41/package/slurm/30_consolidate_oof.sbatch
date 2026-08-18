#!/bin/bash -l
#SBATCH --job-name=alignn-s2-oof
#SBATCH --partition=compute
#SBATCH --account=research-ME-mse
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3968M
#SBATCH --output=logs/oof-%j.out
#SBATCH --error=logs/oof-%j.err
set -euo pipefail
module load 2025
: "${SCRATCH:=/scratch/$USER}"
ENV_PREFIX="${ALIGNN_ENV_PREFIX:-$SCRATCH/conda_envs/alignn_matbench_is_metal_cu118_v9}"
WORK_ROOT="${ALIGNN_WORK_ROOT:-/home/$USER/ml4md/ALIGNN/stage2_primary_v20}"
export ALIGNN_GRAPH_CACHE_ROOT="${ALIGNN_GRAPH_CACHE_ROOT:-$SCRATCH/alignn_matbench_is_metal_graph_cache_v20}"
V26_ROOT="${ALIGNN_V26_PACKAGE_ROOT:-/home/$USER/alignn_stage2_v26/delftblue_package_v26}"
V36_ROOT="${ALIGNN_V36_PACKAGE_ROOT:-/home/$USER/alignn_stage2_v36/delftblue_package_v36}"
OUTPUT_DIR="${ALIGNN_OOF_OUTPUT:-$WORK_ROOT/oof_analysis}"
mkdir -p "$SLURM_SUBMIT_DIR/logs" "$OUTPUT_DIR"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/activate_cuda_runtime.sh
srun bash scripts/require_v37_external_gates.sh "$ENV_PREFIX" "$V26_ROOT" "$V36_ROOT" "$ALIGNN_GRAPH_CACHE_ROOT" "$WORK_ROOT"
srun "$ENV_PREFIX/bin/python" scripts/audit_grid_complete.py --work-root "$WORK_ROOT"
srun "$ENV_PREFIX/bin/python" scripts/consolidate_oof.py --work-root "$WORK_ROOT" --output-dir "$OUTPUT_DIR" --seeds 0 1 2 3 4 --bootstrap-repetitions 5000
echo "ALIGNN_OOF_CONSOLIDATION: PASS"
