# DelftBlue v29 CPU Random2-Coordinate Runbook

## 1. Upload and extract

From Windows PowerShell:

```powershell
scp .\alignn_stage2_delftblue_v29.tar.gz hhew@login.delftblue.tudelft.nl:~/
scp .\DELFTBLUE_ARCHIVE_MANIFEST_V29.json hhew@login.delftblue.tudelft.nl:~/
```

After SSH login:

```bash
mkdir -p "$HOME/alignn_stage2_v29"
tar -xzf "$HOME/alignn_stage2_delftblue_v29.tar.gz" -C "$HOME/alignn_stage2_v29"
cd "$HOME/alignn_stage2_v29/delftblue_package_v29"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
python3 scripts/verify_package.py
python3 scripts/audit_cpu_only.py
mkdir -p logs preflight
bash setup_cpu_environment.sh
"$ALIGNN_ENV_PREFIX/bin/python" -m pytest --assert=plain -q -p no:cacheprovider tests_v29
```

## 2. Simulation-only resource validation

This submits no jobs; reported job IDs are simulations:

```bash
python3 scripts/delftblue_cpu_test_only.py
```

## 3. CPU preflight (permitted while sigma is pending)

```bash
PREFLIGHT_JOB=$(sbatch --parsable slurm/00_cpu_preflight.sbatch)
echo "$PREFLIGHT_JOB"
```

No later scientific job is authorized yet. The immutable `COORDINATE_NOISE_CONFIG.json` must remain null/pending.

## 4. Import prospective sigma authority

The externally produced JSON must contain a finite positive `sigma_cartesian_per_axis_angstrom`, status `approved_prospective_coordinate_sigma`, `scientific_execution_authorized: true`, `selected_prospectively_without_alignn_outcomes: true`, `test_only: false`, this package's aggregate and coordinate-template SHA-256, and a 64-character prospective provenance SHA-256. Do not edit the package configuration.

```bash
export ALIGNN_COORDINATE_SIGMA_SOURCE="$HOME/authorities/SELECTED_COORDINATE_SIGMA.json"
SIGMA_JOB=$(sbatch --parsable --dependency="afterok:$PREFLIGHT_JOB" slurm/01_import_verify_coordinate_sigma_cpu.sbatch)
echo "$SIGMA_JOB"
```

The central gate used by cache, training, inference, calibration, OOF and audit is:

```bash
export ALIGNN_COORDINATE_SIGMA_AUTHORIZATION="$PWD/preflight/VERIFIED_COORDINATE_SIGMA_AUTHORIZATION.json"
python3 -c 'from alignn_stage2.sigma_authorization import require_authorized_sigma; print(require_authorized_sigma())'
```

With no approved artifact, this raises `CoordinateSigmaAuthorizationMissing` before scientific labels, structures, caches or outer-test artifacts are read.

## 5. Coordinate cache, smoke and profile

```bash
CACHE_JOB=$(sbatch --parsable --dependency="afterok:$SIGMA_JOB" slurm/05_build_coordinate_warmup_cache_cpu.sbatch)
SMOKE_JOB=$(sbatch --parsable --dependency="afterok:$CACHE_JOB" slurm/06_smoke_coordinate_fold0_seed0_cpu.sbatch)
PROFILE_JOB=$(sbatch --parsable --dependency="afterok:$SMOKE_JOB" slurm/07_profile_coordinate_100_batches_cpu.sbatch)
printf 'cache=%s smoke=%s profile=%s\n' "$CACHE_JOB" "$SMOKE_JOB" "$PROFILE_JOB"
```

Inspect the profile, then explicitly approve it on the login node:

```bash
python3 -m json.tool preflight/CPU_COORDINATE_100_BATCH_PROFILE.json | less
python3 scripts/review_cpu_profile.py --profile preflight/CPU_COORDINATE_100_BATCH_PROFILE.json --output preflight/CPU_COORDINATE_PROFILE_APPROVAL.json --reviewer "$USER" --approve
python3 scripts/require_cpu_profile.py --profile preflight/CPU_COORDINATE_100_BATCH_PROFILE.json --approval preflight/CPU_COORDINATE_PROFILE_APPROVAL.json
```

## 6. Primary and remaining cells

```bash
PRIMARY_JOB=$(sbatch --parsable slurm/08_primary_coordinate_fold0_seed0_cpu.sbatch)
GRID_JOB=$(sbatch --parsable --dependency="afterok:$PRIMARY_JOB" slurm/10_train_coordinate_fold_seed_cpu.sbatch)
printf 'primary=%s grid=%s\n' "$PRIMARY_JOB" "$GRID_JOB"
```

If a cell returns 75 after its wall-time signal, it is intentionally incomplete. Resubmit that exact cell; its batch-position checkpoint is reused. For example, cell 7:

```bash
sbatch --array=7-7 slurm/10_train_coordinate_fold_seed_cpu.sbatch
```

Do not launch calibration until all 25 cells have no `INCOMPLETE_RESUME_REQUIRED.json` and both selected checkpoints exist. Failed array dependencies must be recreated explicitly after the failed cells are completed.

## 7. Calibration, OOF and final audit

```bash
EXPORT_JOB=$(sbatch --parsable --dependency="afterok:$GRID_JOB" slurm/20_calibrate_export_coordinate_cpu.sbatch)
OOF_JOB=$(sbatch --parsable --dependency="afterok:$EXPORT_JOB" slurm/30_consolidate_coordinate_oof_cpu.sbatch)
AUDIT_JOB=$(sbatch --parsable --dependency="afterok:$OOF_JOB" slurm/40_final_audit_coordinate_cpu.sbatch)
printf 'export=%s oof=%s audit=%s\n' "$EXPORT_JOB" "$OOF_JOB" "$AUDIT_JOB"
```

Monitor with `squeue -u "$USER"` and inspect completion with `sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem`.

The ORB repository evidence freezes the perturbation rule and candidate grid but does not contain a completed prospectively approved sigma selection. Therefore v28 supplies the safe authority-import job rather than inventing a selector or value.
