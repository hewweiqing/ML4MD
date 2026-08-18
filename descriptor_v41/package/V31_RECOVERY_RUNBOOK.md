# DelftBlue v31 calibration-recovery runbook — superseded

Do not execute this inherited v31 runbook from v32. Real DelftBlue validation rejected the v31 `gpu-a100-small` request. Use `V32_RECOVERY_RUNBOOK.md`.

v31 reuses the already trained v26 fold-0/seed-0 staging cell. It does not rerun training. The recovery job first performs validation-only fitting and refuses to proceed if either branch fails convergence or if an outer-test sentinel already exists. Only then does the existing export pipeline cross its outer-test boundary.

## Windows PowerShell upload

From the local directory containing the archive:

```powershell
scp .\alignn_stage2_delftblue_v31.tar.gz hhew@login.delftblue.tudelft.nl:~/
```

## DelftBlue extraction and package verification

```bash
cd "$HOME"
sha256sum alignn_stage2_delftblue_v31.tar.gz
# Compare the printed digest with DELFTBLUE_ARCHIVE_MANIFEST_V31.json supplied beside the archive.
mkdir -p "$HOME/alignn_stage2_v31"
tar -xzf alignn_stage2_delftblue_v31.tar.gz -C "$HOME/alignn_stage2_v31"
cd "$HOME/alignn_stage2_v31/delftblue_package_v31"
export ALIGNN_ENV_PREFIX="/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9"
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 \
  "$ALIGNN_ENV_PREFIX/bin/python" scripts/verify_package.py
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 \
  "$ALIGNN_ENV_PREFIX/bin/python" scripts/audit_slurm_resources.py
mkdir -p logs preflight
```

Do not run `setup_environment.sh`; v31 reuses the already certified environment without changing it.

## Simulation-only DelftBlue check

This command submits no job. Any displayed job ID is a simulation:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 \
  "$ALIGNN_ENV_PREFIX/bin/python" scripts/delftblue_test_only.py
```

## Recovery submission

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
printf '%s\n' "$RECOVERY_JOB" > "$HOME/v31_recovery_job.txt"
echo "RECOVERY_JOB=$RECOVERY_JOB"
```

Monitor with:

```bash
squeue -j "$RECOVERY_JOB"
sacct -X -j "$RECOVERY_JOB" --format=JobID,JobName%32,State,ExitCode,Elapsed,MaxRSS,ReqMem
tail -n 300 "logs/ts-recovery-f0s0-${RECOVERY_JOB}.out"
tail -n 200 "logs/ts-recovery-f0s0-${RECOVERY_JOB}.err"
```

Required final lines are `V31_CALIBRATION_RECOVERY_VALIDATION: PASS`, `ALIGNN_CELL: PASS`, and `ALIGNN_V31_F0S0_CALIBRATION_RECOVERY: PASS`. Do not submit the remaining 24 cells until all three are present and the Slurm state is `COMPLETED` with exit code `0:0`.
