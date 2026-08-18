# DelftBlue GPU Coordinate Experiment Runbook

This additive package moves only the coordinate-intervention experiment from CPU to A100 CUDA. It does not alter the fixed 0.040 Å per-axis dose, official folds, five seeds, model, optimizer, epochs, graph settings, Random2 warm-up, temperature scaler, or analysis. The partial CPU checkpoints are retained as runtime evidence and are never imported.

## 1. Upload from Windows PowerShell

```powershell
$UserName = "hhew"
$Archive = "C:\Users\User\OneDrive - Delft University of Technology\Master Y1\Q4\Research\experiments\matbench_alignn_is_metal_calibration\stage2\alignn_coordinate_gpu_delftblue_v4.tar.gz"
$Manifest = "C:\Users\User\OneDrive - Delft University of Technology\Master Y1\Q4\Research\experiments\matbench_alignn_is_metal_calibration\stage2\DELFTBLUE_COORDINATE_GPU_ARCHIVE_MANIFEST_V4.json"
scp $Archive "${UserName}@login.delftblue.tudelft.nl:~/"
scp $Manifest "${UserName}@login.delftblue.tudelft.nl:~/"
```

## 2. Extract and verify on DelftBlue

```bash
cd "$HOME"
echo "$(python3 -c 'import json; print(json.load(open("DELFTBLUE_COORDINATE_GPU_ARCHIVE_MANIFEST_V4.json"))["archive_sha256"])')  alignn_coordinate_gpu_delftblue_v4.tar.gz" | sha256sum --check
mkdir -p "$HOME/alignn_coordinate_gpu_v4"
tar -xzf alignn_coordinate_gpu_delftblue_v4.tar.gz -C "$HOME/alignn_coordinate_gpu_v4"
cd "$HOME/alignn_coordinate_gpu_v4/delftblue_coordinate_gpu_v4"
mkdir -p logs preflight

export ALIGNN_ENV_PREFIX="/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9"
export ALIGNN_CLEAN_INSTALL_EVIDENCE="$ALIGNN_ENV_PREFIX/.alignn_stage2_v22_clean_install_evidence.json"
export ALIGNN_CLEAN_INSTALL_TRANSCRIPT="$ALIGNN_ENV_PREFIX/.alignn_stage2_v22_clean_install_transcript.json"
export DGLBACKEND=pytorch
export LD_LIBRARY_PATH="$ALIGNN_ENV_PREFIX/lib:$ALIGNN_ENV_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$ALIGNN_ENV_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$ALIGNN_ENV_PREFIX/lib/python3.10/site-packages/nvidia/cusolver/lib:$ALIGNN_ENV_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 \
  "$ALIGNN_ENV_PREFIX/bin/python" scripts/verify_package.py
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 \
  "$ALIGNN_ENV_PREFIX/bin/python" -m pytest -q -p no:cacheprovider tests_gpu
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 \
  "$ALIGNN_ENV_PREFIX/bin/python" scripts/audit_gpu_coordinate_resources.py
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 \
  "$ALIGNN_ENV_PREFIX/bin/python" scripts/delftblue_test_only.py
```

`sbatch --test-only` is simulation only; the displayed job IDs are not submitted jobs.

## 3. Define the isolated GPU-coordinate roots

```bash
export ALIGNN_DATASET="/scratch/$USER/alignn_matbench_is_metal_data/matbench_mp_is_metal.json.gz"
export ALIGNN_GRAPH_CACHE_ROOT="/scratch/$USER/alignn_matbench_is_metal_graph_cache_v26_verified"
export ALIGNN_COORDINATE_CACHE_ROOT="/scratch/$USER/alignn_random2_coordinate_cache_v29"
export ALIGNN_V29_ROOT="$HOME/alignn_stage2_v29/delftblue_package_v29"
export ALIGNN_COORDINATE_SIGMA_AUTHORIZATION="$ALIGNN_V29_ROOT/preflight/VERIFIED_COORDINATE_SIGMA_AUTHORIZATION.json"
export ALIGNN_GPU_COORDINATE_WORK_ROOT="$HOME/ml4md/ALIGNN/gpu_coordinate_primary_v4"
export ALIGNN_GPU_COORDINATE_OOF_ROOT="$HOME/ml4md/ALIGNN/gpu_coordinate_oof_v4"

# The external authority remains the byte-identical v29 artifact. v2 verifies
# its SHA-256 and v29 origin package lineage; do not rewrite it for this package.
```

Do not set `ALIGNN_GPU_COORDINATE_WORK_ROOT` to the CPU coordinate root. The package refuses such a path. Do not resume the old CPU job after adopting this prospective device amendment.

## 4. Required outcome-neutral A100 gates

Run one job at a time and inspect its exit state and named PASS line before continuing.

```bash
PREFLIGHT_JOB=$(sbatch --parsable slurm/00_gpu_coordinate_a100_preflight.sbatch); PREFLIGHT_JOB="${PREFLIGHT_JOB%%;*}"
echo "$PREFLIGHT_JOB" > "$HOME/gpu_coordinate_v4_preflight_job.txt"
```

After `COMPLETED 0:0` and `GPU_COORDINATE_A100_PREFLIGHT: PASS`:

```bash
SMOKE_JOB=$(sbatch --parsable --export=ALL slurm/06_smoke_coordinate_f0s0_gpu.sbatch); SMOKE_JOB="${SMOKE_JOB%%;*}"
echo "$SMOKE_JOB" > "$HOME/gpu_coordinate_v4_smoke_job.txt"
```

After `COMPLETED 0:0` and `ALIGNN_GPU_COORDINATE_NON_PRIMARY_SMOKE: PASS`:

```bash
PROFILE_JOB=$(sbatch --parsable --export=ALL slurm/07_profile_coordinate_100_batches_gpu.sbatch); PROFILE_JOB="${PROFILE_JOB%%;*}"
echo "$PROFILE_JOB" > "$HOME/gpu_coordinate_v4_profile_job.txt"
```

After `COMPLETED 0:0` and `ALIGNN_COORDINATE_GPU_100_BATCH_PROFILE: PASS`, review the outcome-neutral resource report:

```bash
"$ALIGNN_ENV_PREFIX/bin/python" -m json.tool preflight/GPU_COORDINATE_100_BATCH_PROFILE.json
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" "$ALIGNN_ENV_PREFIX/bin/python" scripts/review_gpu_coordinate_profile.py \
  --profile preflight/GPU_COORDINATE_100_BATCH_PROFILE.json \
  --output preflight/GPU_COORDINATE_PROFILE_APPROVAL.json \
  --reviewer "$USER@$(hostname)" --decision approve \
  --reason "100 representative A100 optimizer batches passed; projected runtime and peak CUDA reservation satisfy the prespecified GPU policy."
```

If any gate fails, stop. Do not substitute another sigma and do not inspect coordinate outer-test results.

## 5. Primary fold 0 / seed 0

```bash
PRIMARY_JOB=$(sbatch --parsable --export=ALL slurm/08_primary_coordinate_f0s0_gpu.sbatch); PRIMARY_JOB="${PRIMARY_JOB%%;*}"
echo "$PRIMARY_JOB" > "$HOME/gpu_coordinate_v4_primary_job.txt"
```

If it exits 75, the checkpoint is deliberate and resumable; submit the exact same command again. After `COMPLETED 0:0`:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" "$ALIGNN_ENV_PREFIX/bin/python" scripts/write_gpu_coordinate_primary_gate.py \
  --work-root "$ALIGNN_GPU_COORDINATE_WORK_ROOT" \
  --output preflight/GPU_COORDINATE_PRIMARY_FOLD0_GATE.json
```

## 6. Remaining training, calibration/export, OOF, and audit

Submit the remaining 24 training cells only after the primary gate exists:

```bash
TRAIN_JOB=$(sbatch --parsable --export=ALL slurm/10_train_coordinate_grid_gpu.sbatch); TRAIN_JOB="${TRAIN_JOB%%;*}"
echo "$TRAIN_JOB" > "$HOME/gpu_coordinate_v4_training_job.txt"
CAL_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB --export=ALL slurm/20_calibrate_export_coordinate_gpu.sbatch); CAL_JOB="${CAL_JOB%%;*}"
echo "$CAL_JOB" > "$HOME/gpu_coordinate_v4_calibration_job.txt"
OOF_JOB=$(sbatch --parsable --dependency=afterok:$CAL_JOB --export=ALL slurm/30_consolidate_coordinate_oof_gpu.sbatch); OOF_JOB="${OOF_JOB%%;*}"
echo "$OOF_JOB" > "$HOME/gpu_coordinate_v4_oof_job.txt"
AUDIT_JOB=$(sbatch --parsable --dependency=afterok:$OOF_JOB --export=ALL slurm/40_final_audit_coordinate_gpu.sbatch); AUDIT_JOB="${AUDIT_JOB%%;*}"
echo "$AUDIT_JOB" > "$HOME/gpu_coordinate_v4_audit_job.txt"
```

If a training array element exits 75, its `afterok` dependency will remain unsatisfied. Resubmit only that array index with `sbatch --array=<index> --export=ALL slurm/10_train_coordinate_grid_gpu.sbatch`, verify all 25 `TRAINING_STATUS.json` files, then submit a fresh calibration/OOF/audit chain. Never release downstream jobs from partial cells.

## Monitoring

```bash
squeue -u "$USER"
sacct -X -j "$JOB_ID" --format=JobID,JobName%36,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocTRES%60
```

The existing descriptor GPU array and this coordinate GPU workflow compete for the same A100 allocation. For predictable scheduling, let the descriptor array finish or reduce concurrent coordinate tasks; this affects queueing only, not scientific independence.
