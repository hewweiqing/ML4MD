# DelftBlue v33 calibration-recovery runbook — superseded

Use `V34_RECOVERY_RUNBOOK.md`. v34 changes only the inherited expected count of regular `gpu-a100` scripts from seven to eight; operational and scientific behavior is unchanged.

v33 reuses the completed v26 fold-0/seed-0 checkpoints and validation logits. It first repeats validation-only fitting and refuses an existing outer-test sentinel. It then runs the inherited calibration/export, verification and atomic-promotion path. It does not retrain the cell.

```bash
cd "$HOME/alignn_stage2_v33/delftblue_package_v33"
export ALIGNN_ENV_PREFIX="/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9"
export DGLBACKEND=pytorch
source scripts/activate_cuda_runtime.sh
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 \
  "$ALIGNN_ENV_PREFIX/bin/python" scripts/verify_package.py
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 \
  "$ALIGNN_ENV_PREFIX/bin/python" scripts/audit_slurm_resources.py
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 \
  "$ALIGNN_ENV_PREFIX/bin/python" scripts/delftblue_test_only.py
```

Only after all three checks pass:

```bash
export ALIGNN_V26_PACKAGE_ROOT="$HOME/alignn_stage2_v26/delftblue_package_v26"
export ALIGNN_GRAPH_CACHE_ROOT="/scratch/$USER/alignn_matbench_is_metal_graph_cache_v26_verified"
export ALIGNN_STAGING_ROOT="$HOME/ml4md/ALIGNN/.stage2_primary_v26_staging"
export ALIGNN_WORK_ROOT="$HOME/ml4md/ALIGNN/stage2_primary_v26"
export ALIGNN_QUARANTINE_ROOT="$HOME/ml4md/ALIGNN/quarantine_v26"
RECOVERY_SUBMISSION=$(sbatch --parsable \
  --export=ALL,ALIGNN_ENV_PREFIX="$ALIGNN_ENV_PREFIX",ALIGNN_V26_PACKAGE_ROOT="$ALIGNN_V26_PACKAGE_ROOT",ALIGNN_GRAPH_CACHE_ROOT="$ALIGNN_GRAPH_CACHE_ROOT",ALIGNN_STAGING_ROOT="$ALIGNN_STAGING_ROOT",ALIGNN_WORK_ROOT="$ALIGNN_WORK_ROOT",ALIGNN_QUARANTINE_ROOT="$ALIGNN_QUARANTINE_ROOT" \
  slurm/09_recover_calibration_f0s0.sbatch)
RECOVERY_JOB="${RECOVERY_SUBMISSION%%;*}"
test -n "$RECOVERY_JOB" || { echo "STOP: no recovery job ID"; exit 1; }
printf '%s\n' "$RECOVERY_JOB" > "$HOME/v33_recovery_job.txt"
echo "RECOVERY_JOB=$RECOVERY_JOB"
```
