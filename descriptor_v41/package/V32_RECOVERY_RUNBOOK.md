# DelftBlue v32 calibration-recovery runbook — superseded

Do not execute this inherited v32 runbook from v33. DelftBlue job `10643036` exposed the exact-ordering assertion corrected in v33. Use `V33_RECOVERY_RUNBOOK.md`.

v32 supersedes v31 operationally because live test-only validation showed that `gpu-a100-small` cannot accept eight CPUs per task. The v31 scientific numerical amendment remains unchanged. The recovery job runs on regular `gpu-a100` for three hours and does not retrain fold 0/seed 0.

After upload and extraction, reuse the certified environment and verify the package:

```bash
cd "$HOME/alignn_stage2_v32/delftblue_package_v32"
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

The final command is simulation-only and must report `jobs_submitted: 0` and `status: passed` before recovery submission.

Submit only after test-only passes:

```bash
export ALIGNN_V26_PACKAGE_ROOT="$HOME/alignn_stage2_v26/delftblue_package_v26"
export ALIGNN_GRAPH_CACHE_ROOT="/scratch/$USER/alignn_matbench_is_metal_graph_cache_v26_verified"
export ALIGNN_STAGING_ROOT="$HOME/ml4md/ALIGNN/.stage2_primary_v26_staging"
export ALIGNN_WORK_ROOT="$HOME/ml4md/ALIGNN/stage2_primary_v26"
export ALIGNN_QUARANTINE_ROOT="$HOME/ml4md/ALIGNN/quarantine_v26"
RECOVERY_SUBMISSION=$(sbatch --parsable \
  --export=ALL,ALIGNN_V26_PACKAGE_ROOT="$ALIGNN_V26_PACKAGE_ROOT",ALIGNN_GRAPH_CACHE_ROOT="$ALIGNN_GRAPH_CACHE_ROOT",ALIGNN_STAGING_ROOT="$ALIGNN_STAGING_ROOT",ALIGNN_WORK_ROOT="$ALIGNN_WORK_ROOT",ALIGNN_QUARANTINE_ROOT="$ALIGNN_QUARANTINE_ROOT" \
  slurm/09_recover_calibration_f0s0.sbatch)
RECOVERY_JOB="${RECOVERY_SUBMISSION%%;*}"
test -n "$RECOVERY_JOB" || { echo "STOP: no recovery job ID"; exit 1; }
printf '%s\n' "$RECOVERY_JOB" > "$HOME/v32_recovery_job.txt"
echo "RECOVERY_JOB=$RECOVERY_JOB"
```
