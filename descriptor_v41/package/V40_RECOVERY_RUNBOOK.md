# DelftBlue v40 numerical-resolution recovery runbook

v40 is an additive recovery for array indices 6, 14 and 16 only. It does not retrain a model. It first produces a validation-only resolution record, then performs each cell's first and only outer-test export, verifies the cell and atomically promotes it.

## Upload from Windows PowerShell

```powershell
$Base = "C:\Users\User\OneDrive - Delft University of Technology\Master Y1\Q4\Research\experiments\matbench_alignn_is_metal_calibration\stage2"
scp "$Base\alignn_stage2_delftblue_v40.tar.gz" "hhew@login.delftblue.tudelft.nl:~/"
scp "$Base\DELFTBLUE_ARCHIVE_MANIFEST_V40.json" "hhew@login.delftblue.tudelft.nl:~/"
```

## Extract and validate

```bash
cd "$HOME"
echo "$(python3 -c 'import json; print(json.load(open("DELFTBLUE_ARCHIVE_MANIFEST_V40.json"))["archive_sha256"])')  alignn_stage2_delftblue_v40.tar.gz" | sha256sum --check
mkdir -p "$HOME/alignn_stage2_v40"
tar -xzf alignn_stage2_delftblue_v40.tar.gz -C "$HOME/alignn_stage2_v40"
cd "$HOME/alignn_stage2_v40/delftblue_package_v40"
mkdir -p logs preflight

export ALIGNN_ENV_PREFIX="/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9"
export ALIGNN_CLEAN_INSTALL_EVIDENCE="$ALIGNN_ENV_PREFIX/.alignn_stage2_v22_clean_install_evidence.json"
export ALIGNN_CLEAN_INSTALL_TRANSCRIPT="$ALIGNN_ENV_PREFIX/.alignn_stage2_v22_clean_install_transcript.json"
export DGLBACKEND=pytorch
source scripts/activate_cuda_runtime.sh

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 \
  "$ALIGNN_ENV_PREFIX/bin/python" scripts/verify_package.py
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 \
  "$ALIGNN_ENV_PREFIX/bin/python" -m pytest --assert=plain -q -p no:cacheprovider tests
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 \
  "$ALIGNN_ENV_PREFIX/bin/python" scripts/delftblue_test_only.py
```

## Export roots and simulate the exact recovery job

```bash
export ALIGNN_V26_PACKAGE_ROOT="$HOME/alignn_stage2_v26/delftblue_package_v26"
export ALIGNN_V36_PACKAGE_ROOT="$HOME/alignn_stage2_v36/delftblue_package_v36"
export ALIGNN_GRAPH_CACHE_ROOT="/scratch/$USER/alignn_matbench_is_metal_graph_cache_v26_verified"
export ALIGNN_STAGING_ROOT="$HOME/ml4md/ALIGNN/.stage2_primary_v26_staging"
export ALIGNN_WORK_ROOT="$HOME/ml4md/ALIGNN/stage2_primary_v26"
export ALIGNN_QUARANTINE_ROOT="$HOME/ml4md/ALIGNN/quarantine_v26"

sbatch --test-only slurm/13_resolve_export_preouter_v40.sbatch
```

The test-only job ID is a simulation and submits no job.

## Submit once

```bash
V40_SUBMISSION=$(sbatch --parsable slurm/13_resolve_export_preouter_v40.sbatch)
V40_JOB="${V40_SUBMISSION%%;*}"
test -n "$V40_JOB" || { echo "STOP: no v40 job ID returned"; exit 1; }
printf '%s\n' "$V40_JOB" > "$HOME/v40_resolution_export_job.txt"
echo "V40_JOB=$V40_JOB"
squeue -r -j "$V40_JOB"
```

Do not submit this array twice. Do not submit OOF consolidation until all three tasks complete, each final `COMPLETE.json` exists, and the full-grid audit passes.
