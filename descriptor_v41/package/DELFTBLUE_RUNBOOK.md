# DelftBlue v35 operator runbook

> **v35 recovery notice:** DelftBlue job `10644192` already performed its single outer-test export and then failed in the inherited exact production-ordering verifier. Do not rerun an older calibration/export package. Follow `V35_RECOVERY_RUNBOOK.md`; v35 validates and reuses the existing artifacts without rematerialization.

## Preserved v26 operational history

v26 removes the single false-negative v25 substring assertion after DelftBlue runtime evidence showed `143 passed, 1 failed`. The AST-based profiler shadowing regression remains. Use `~/alignn_stage2_v26/delftblue_package_v26`; do not modify or rerun v25.

## v25 additive profiler correction

v25 fixed only the report-path/model-tensor name collision observed after the representative batches in profile job `10633397`. v26 retains that correction and fixes only its false-negative test. Do not approve the failed v24 profile, which produced no report.

After entering the v26 directory, explicitly select v26 runtime locations:

```bash
export ALIGNN_ENV_PREFIX="/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9"
export ALIGNN_WORK_ROOT="/home/$USER/ml4md/ALIGNN/stage2_primary_v26"
export ALIGNN_STAGING_ROOT="/home/$USER/ml4md/ALIGNN/.stage2_primary_v26_staging"
export ALIGNN_QUARANTINE_ROOT="/home/$USER/ml4md/ALIGNN/quarantine_v26"
export ALIGNN_GRAPH_CACHE_ROOT="/scratch/$USER/alignn_matbench_is_metal_graph_cache_v26"
export ALIGNN_CACHE_QUARANTINE_ROOT="/scratch/$USER/alignn_matbench_is_metal_graph_cache_quarantine_v26"
export ALIGNN_SMOKE_ROOT="/home/$USER/ml4md/ALIGNN/non_primary_smoke_v26"
```

Run `bash setup_environment.sh` once to revalidate the unchanged environment and bind its login certification to v26. Then repeat the package-bound A100, cache, and smoke gates before submitting `slurm/07_profile_100_batches.sbatch`. Primary training remains prohibited until the resulting v26 profile is manually reviewed and approved.

Replace `<netid>` only. Every runtime gate is pending in the distributed archive. Run from a DelftBlue login shell unless stated otherwise.

## Upload, verify and extract

Windows PowerShell:

```powershell
scp .\alignn_stage2_delftblue_v20.tar.gz .\DELFTBLUE_ARCHIVE_MANIFEST_V20.json <netid>@login.delftblue.tudelft.nl:~/
```

DelftBlue:

```bash
ssh <netid>@login.delftblue.tudelft.nl
cd ~
python3 -c 'import hashlib,json,pathlib; m=json.loads(pathlib.Path("DELFTBLUE_ARCHIVE_MANIFEST_V20.json").read_text()); p=pathlib.Path(m["archive"]); assert hashlib.sha256(p.read_bytes()).hexdigest()==m["archive_sha256"]; print("ARCHIVE_SHA256: PASS")'
mkdir -p ~/alignn_stage2_v20
tar -xzf ~/alignn_stage2_delftblue_v20.tar.gz -C ~/alignn_stage2_v20
cd ~/alignn_stage2_v26/delftblue_package_v26
export ALIGNN_ENV_PREFIX="/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9"
export ALIGNN_WORK_ROOT="/home/$USER/ml4md/ALIGNN/stage2_primary_v26"
export ALIGNN_STAGING_ROOT="/home/$USER/ml4md/ALIGNN/.stage2_primary_v26_staging"
export ALIGNN_QUARANTINE_ROOT="/home/$USER/ml4md/ALIGNN/quarantine_v26"
export ALIGNN_GRAPH_CACHE_ROOT="/scratch/$USER/alignn_matbench_is_metal_graph_cache_v26_verified"
export ALIGNN_V26_PACKAGE_ROOT="$HOME/alignn_stage2_v26/delftblue_package_v26"
export ALIGNN_V36_PACKAGE_ROOT="$HOME/alignn_stage2_v36/delftblue_package_v36"
export ALIGNN_CACHE_QUARANTINE_ROOT="/scratch/$USER/alignn_matbench_is_metal_graph_cache_quarantine_v20"
(
  module purge
  module load 2024r1
  module load python/3.10.12
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/verify_package.py
)
command -v sbatch
sbatch --version
bash setup_environment.sh
mkdir -p logs preflight
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
PYTHONDONTWRITEBYTECODE=1 \
"$ALIGNN_ENV_PREFIX/bin/python" scripts/audit_slurm_resources.py
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
PYTHONDONTWRITEBYTECODE=1 \
"$ALIGNN_ENV_PREFIX/bin/python" scripts/delftblue_test_only.py
```

The default DelftBlue login interpreter is Python 3.6.8 and cannot parse `from __future__ import annotations` (`future feature annotations is not defined`). Bootstrap verification therefore uses Python 3.10.12 inside the parentheses. You must not run `module purge` directly in the long-lived parent login shell: the isolated subshell returns without removing the parent's Slurm commands. `command -v sbatch` and `sbatch --version` must both succeed before any test-only or submitted job.

If Conda is not on `PATH`, `setup_environment.sh` checks `$HOME/miniforge3/bin/conda`; an executable `CONDA_EXE` takes precedence. It creates Python 3.10.20 in a new `cu118_v9.staging` prefix, pins pip itself to 25.3, and performs a Conda prefix-aware clone only after hash-locked installs, loader/import checks, `pip check`, exact installed-closure verification, package verification and bootstrap-safe tests pass. It then verifies the final clone, writes runtime evidence beside that final prefix, runs the complete test suite against that external evidence, and only then atomically writes the login certification. This ordering removes the v17 circular gate: the immutable package contains static resolution proof and a deliberately pending runtime placeholder, while the real clean-install evidence is generated by setup. A partial final prefix or stale staging prefix is rejected. An existing Python executable is not certification.

The revision-9 environment uses the verified hash locks in `BOOTSTRAP_REQUIREMENTS.txt`, `TORCH_CU118_REQUIREMENT.txt`, `DGL_CU118_REQUIREMENT.txt`, `CUDA11_RUNTIME_REQUIREMENTS.txt`, and `PYTHON_DEPENDENCY_LOCK.txt`. Every lock installation uses `--no-deps --require-hashes`; therefore pip cannot resolve an undeclared distribution. The Python 3.10-only pytest dependencies `exceptiongroup==1.3.1` and `tomli==2.4.1` are explicitly hash-pinned. `pip check` then proves graph satisfaction. CUDA-11 cuSPARSE has no nvJitLink dependency. Login certification loads `libcudart.so.11.0`, `libcublas.so.11`, `libcusparse.so.11`, and `libcusolver.so.11`, then imports torch, DGL and ALIGNN without using a GPU.

Inspect the immutable placeholder, static proof, and generated runtime proof without modifying them:

```bash
python3 -m json.tool CLEAN_INSTALL_EVIDENCE.json | less
python3 -m json.tool DEPENDENCY_RESOLUTION_EVIDENCE.json | less
python3 -m json.tool "$ALIGNN_ENV_PREFIX/.alignn_stage2_v20_clean_install_evidence.json" | less
python3 -m json.tool "$ALIGNN_ENV_PREFIX/.alignn_stage2_v20_clean_install_transcript.json" | less
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
PYTHONDONTWRITEBYTECODE=1 \
"$ALIGNN_ENV_PREFIX/bin/python" scripts/verify_dependency_resolution.py
"$ALIGNN_ENV_PREFIX/bin/python" -m pip check
cat "$ALIGNN_ENV_PREFIX/.alignn_stage2_v9_login_certification.json"
```

The exact observed v19 final prefix is recoverable without deletion. Preserve its diagnostic and `.alignn_stage2_v19_environment_report.provisional.json`, extract v20, and run `bash setup_environment.sh`. v20 validates that report before replaying only the hash-locked bootstrap tools. Do not run the cleanup block for that exact state.

For any other inspected and explicitly rejected uncertified prefix, cleanup remains:

```bash
chmod -R u+w "$ALIGNN_ENV_PREFIX" 2>/dev/null || true
rm -rf "$ALIGNN_ENV_PREFIX"
rm -rf "${ALIGNN_ENV_PREFIX}.staging"
bash setup_environment.sh
```

The live helper invokes `sbatch --test-only` for every `.sbatch` file. Any printed test-only job ID is a simulation; the helper submits zero jobs. It fails on a nonzero return code, `error:`, `allocation failure`, or the per-node/non-exclusive warning.

The exact first DelftBlue test-only command after bootstrap and environment verification is:

```bash
cd ~/alignn_stage2_v26/delftblue_package_v26 && PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 "$ALIGNN_ENV_PREFIX/bin/python" scripts/delftblue_test_only.py
```

## Stage 0: A100 runtime certification

The exact first DelftBlue job command is:

```bash
cd ~/alignn_stage2_v26/delftblue_package_v26 && mkdir -p logs preflight && sbatch slurm/00_a100_preflight.sbatch
```

Wait for completion and require `ALIGNN_ENVIRONMENT: PASS`:

```bash
sacct -j <preflight-jobid> --format=JobID,State,Elapsed,MaxRSS,ReqMem,AllocTRES%60
tail -n 100 logs/preflight-<preflight-jobid>.out
tail -n 100 logs/preflight-<preflight-jobid>.err
```

## Stage 1: label-free cache construction and exhaustive verification

Submit only after inspecting Stage 0:

```bash
CACHE_JOB=$(sbatch --parsable slurm/05_build_graph_cache.sbatch)
sacct -j "$CACHE_JOB" --format=JobID,State,Elapsed,MaxRSS,ExitCode
tail -n 100 "logs/cache-$CACHE_JOB.out"
```

Require `ALIGNN_GRAPH_CACHE: PASS`. Invalid, partial or stale shard evidence is moved to the explicit cache-quarantine root. It is never silently deleted. A PASS manifest is promoted only after exhaustive verification.

## Stage 2a: smoke, then manual inspection

The smoke first runs `scripts/verify_execution_contracts.py --device cuda` on synthetic tensors only. Require its JSON `status: passed`; it covers the singleton final batch, pre-LogSoftmax loss identity, uninterrupted/resumed equivalence, exact OneCycle steps, paired initial hashes, BatchNorm-state immutability, CPU/CUDA RNG restoration, sample-order generator restoration, and ranking/class/AUROC invariance. It then runs the existing non-primary fold-0/seed-0 smoke. Neither part authorizes primary training.

```bash
SMOKE_JOB=$(sbatch --parsable slurm/06_smoke_fold0_seed0.sbatch)
sacct -j "$SMOKE_JOB" --format=JobID,State,Elapsed,MaxRSS,ExitCode
tail -n 200 "logs/smoke-f0s0-$SMOKE_JOB.out"
tail -n 100 "logs/smoke-f0s0-$SMOKE_JOB.err"
```

Do not submit the profile until smoke output has been inspected and passed.

## Stage 2b: resource profile, inspection and explicit decision

Submit the profile manually; it does not submit or depend automatically into the primary job:

```bash
PROFILE_JOB=$(sbatch --parsable slurm/07_profile_100_batches.sbatch)
sacct -j "$PROFILE_JOB" --format=JobID,State,Elapsed,MaxRSS,ReqMem,AllocTRES%60
tail -n 200 "logs/profile100-$PROFILE_JOB.out"
tail -n 100 "logs/profile100-$PROFILE_JOB.err"
python3 -m json.tool preflight/FULL_CONFIG_100_BATCH_PROFILE.json | less
```

Review cache timings, both branch projections, conservative upper bound, host RSS, CUDA allocated/reserved/total memory, disk capacity and every included/excluded phase. To approve after all four 80% margin checks pass:

```bash
"$ALIGNN_ENV_PREFIX/bin/python" scripts/review_profile.py \
  --profile preflight/FULL_CONFIG_100_BATCH_PROFILE.json \
  --certification preflight/A100_RUNTIME_CERTIFICATION.json \
  --package-manifest PACKAGE_MANIFEST.json \
  --execution-plan STAGE_2_EXECUTION_PLAN.json \
  --cache-manifest "$ALIGNN_GRAPH_CACHE_ROOT/STRUCTURE_CACHE_MANIFEST.json" \
  --resource-policy RESOURCE_POLICY.json \
  --primary-slurm slurm/08_primary_fold0_seed0.sbatch \
  --output preflight/PROFILE_APPROVAL.json \
  --reviewer "$USER@$(hostname)" --decision approve \
  --reason "Profile inspected; runtime, host memory, CUDA reserved memory and disk margins pass."
```

To record rejection instead:

```bash
set +e
"$ALIGNN_ENV_PREFIX/bin/python" scripts/review_profile.py \
  --profile preflight/FULL_CONFIG_100_BATCH_PROFILE.json \
  --certification preflight/A100_RUNTIME_CERTIFICATION.json \
  --package-manifest PACKAGE_MANIFEST.json \
  --execution-plan STAGE_2_EXECUTION_PLAN.json \
  --cache-manifest "$ALIGNN_GRAPH_CACHE_ROOT/STRUCTURE_CACHE_MANIFEST.json" \
  --resource-policy RESOURCE_POLICY.json \
  --primary-slurm slurm/08_primary_fold0_seed0.sbatch \
  --output preflight/PROFILE_APPROVAL.json \
  --reviewer "$USER@$(hostname)" --decision reject \
  --reason "Specify the failed margin or contradictory measurement."
REVIEW_RC=$?
set -e
test "$REVIEW_RC" -eq 2
```

Before repeating an unchanged v20 profile, preserve prior evidence additively:

```bash
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "preflight/profile_history/$STAMP"
for f in FULL_CONFIG_100_BATCH_PROFILE.json PROFILE_APPROVAL.json; do
  if [ -f "preflight/$f" ]; then mv "preflight/$f" "preflight/profile_history/$STAMP/$f"; fi
done
PROFILE_JOB=$(sbatch --parsable slurm/07_profile_100_batches.sbatch)
```

Changing wall time, CPU count, host memory, GPU request, cache identity, package identity, execution plan or resource policy invalidates approval. Changes to packaged resource files require a new additive release; do not edit a verified v26 extraction in place. A new profile and approval are always required after any permitted runtime evidence change.

## Stage 2c: primary cell—manual submission only

First verify the approval explicitly:

```bash
bash scripts/require_profile_current.sh "$ALIGNN_ENV_PREFIX/bin/python" "$ALIGNN_GRAPH_CACHE_ROOT"
```

Only after that command passes, submit the primary manually. There is intentionally no `afterok:$PROFILE_JOB` dependency:

```bash
PRIMARY_JOB=$(sbatch --parsable slurm/08_primary_fold0_seed0.sbatch)
sacct -j "$PRIMARY_JOB" --format=JobID,State,Elapsed,MaxRSS,ReqMem,AllocTRES%60
tail -n 200 "logs/primary-f0s0-$PRIMARY_JOB.out"
```

Require `ALIGNN_FOLD0_SEED0_PRIMARY: PASS` before the grid.

## Stage 3: paired grid and calibration

```bash
TRAIN_JOB=$(sbatch --parsable slurm/10_train_fold_seed.sbatch)
CAL_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB slurm/20_calibrate_export.sbatch)
sacct -j "$TRAIN_JOB" --array --format=JobIDRaw,State,Elapsed,MaxRSS,ExitCode
```

If a training task fails, cancel only the pending calibration dependency, inspect the evidence and retry only failed indices. Verified complete cells are skipped:

```bash
scancel "$CAL_JOB"
sacct -j "$TRAIN_JOB" --array --format=JobIDRaw,State,ExitCode
RETRY_JOB=$(sbatch --parsable --array=<comma-separated-failed-indices> slurm/10_train_fold_seed.sbatch)
CAL_JOB=$(sbatch --parsable --dependency=afterok:$RETRY_JOB slurm/20_calibrate_export.sbatch)
```

If training passed but calibration/export itself failed, inspect and quarantine only invalid partial cell evidence, then retry only the failed calibration indices without reusing the failed dependency:

```bash
sacct -j "$CAL_JOB" --array --format=JobIDRaw,State,ExitCode
CAL_RETRY_JOB=$(sbatch --parsable --array=<comma-separated-failed-indices> slurm/20_calibrate_export.sbatch)
CAL_JOB="$CAL_RETRY_JOB"
```

For invalid staging/final evidence, inspect first and move only the exact cell path to `$ALIGNN_QUARANTINE_ROOT` before retrying. Never broadly delete runtime evidence. Verify the grid:

```bash
PYTHONPATH="$PWD" "$ALIGNN_ENV_PREFIX/bin/python" scripts/audit_grid_complete.py --work-root "$ALIGNN_WORK_ROOT"
```

## Stage 4: OOF consolidation and final audit

```bash
OOF_JOB=$(sbatch --parsable --dependency=afterok:$CAL_JOB slurm/30_consolidate_oof.sbatch)
AUDIT_JOB=$(sbatch --parsable --dependency=afterok:$OOF_JOB slurm/40_final_audit.sbatch)
sacct -j "$OOF_JOB,$AUDIT_JOB" --format=JobID,State,Elapsed,MaxRSS,ExitCode
tail -n 100 "logs/final-audit-$AUDIT_JOB.out"
```

The final line must be `ALIGNN_PERSISTENT_FINAL_AUDIT: PASS`.
# v40 operational supersession

For the authorized post-training, pre-outer-test numerical resolution of the final three cells, follow `V40_RECOVERY_RUNBOOK.md`. Earlier training, calibration and recovery instructions remain preserved as history and must not be resubmitted for these cells.
