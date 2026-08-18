# DelftBlue v35 existing-export recovery

This release is for the failed v34 fold-0/seed-0 recovery only. It reuses the single existing export under the v26 staging cell and performs no retraining or outer-test rematerialization.

The recovery requests `gpu-a100-small`, two CPUs and one hour because it performs only validation-fit replay, existing-artifact verification and atomic promotion. The two-CPU request complies with the live DelftBlue limit for the small A100 partition.

```bash
cd "$HOME/alignn_stage2_v35/delftblue_package_v35"
export ALIGNN_ENV_PREFIX="/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9"
export ALIGNN_V26_PACKAGE_ROOT="$HOME/alignn_stage2_v26/delftblue_package_v26"
export ALIGNN_GRAPH_CACHE_ROOT="/scratch/$USER/alignn_matbench_is_metal_graph_cache_v26_verified"
export ALIGNN_STAGING_ROOT="$HOME/ml4md/ALIGNN/.stage2_primary_v26_staging"
export ALIGNN_WORK_ROOT="$HOME/ml4md/ALIGNN/stage2_primary_v26"
export ALIGNN_QUARANTINE_ROOT="$HOME/ml4md/ALIGNN/quarantine_v26"
mkdir -p logs preflight

RECOVERY_SUBMISSION=$(sbatch --parsable \
  --export=ALL,ALIGNN_ENV_PREFIX="$ALIGNN_ENV_PREFIX",ALIGNN_V26_PACKAGE_ROOT="$ALIGNN_V26_PACKAGE_ROOT",ALIGNN_GRAPH_CACHE_ROOT="$ALIGNN_GRAPH_CACHE_ROOT",ALIGNN_STAGING_ROOT="$ALIGNN_STAGING_ROOT",ALIGNN_WORK_ROOT="$ALIGNN_WORK_ROOT",ALIGNN_QUARANTINE_ROOT="$ALIGNN_QUARANTINE_ROOT" \
  slurm/09_recover_calibration_f0s0.sbatch)
RECOVERY_JOB="${RECOVERY_SUBMISSION%%;*}"
printf '%s\n' "$RECOVERY_JOB" > "$HOME/v35_recovery_job.txt"
echo "RECOVERY_JOB=$RECOVERY_JOB"
```

Required success lines are:

```text
V35_EXISTING_EXPORT_RECOVERY_VALIDATION: PASS
V35_SINGLE_USE_EXPORT_REUSED: PASS
ALIGNN_CELL: PASS
ALIGNN_V35_F0S0_CALIBRATION_RECOVERY: PASS
```
