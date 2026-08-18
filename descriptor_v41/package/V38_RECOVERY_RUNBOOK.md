# DelftBlue v38 full-grid recovery runbook

Do not resubmit v37 calibration array `10646254`. Do not run OOF consolidation while fewer than 25 final `COMPLETE.json` markers exist.

## Upload from Windows PowerShell

```powershell
$Base = "C:\Users\User\OneDrive - Delft University of Technology\Master Y1\Q4\Research\experiments\matbench_alignn_is_metal_calibration\stage2"
scp "$Base\alignn_stage2_delftblue_v38.tar.gz" "hhew@login.delftblue.tudelft.nl:~/"
scp "$Base\DELFTBLUE_ARCHIVE_MANIFEST_V38.json" "hhew@login.delftblue.tudelft.nl:~/"
```

## Extract and verify

```bash
cd "$HOME"
echo "$(python3 -c 'import json; print(json.load(open("DELFTBLUE_ARCHIVE_MANIFEST_V38.json"))["archive_sha256"])')  alignn_stage2_delftblue_v38.tar.gz" | sha256sum --check
mkdir -p "$HOME/alignn_stage2_v38"
tar -xzf alignn_stage2_delftblue_v38.tar.gz -C "$HOME/alignn_stage2_v38"
cd "$HOME/alignn_stage2_v38/delftblue_package_v38"
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

The last command is simulation-only and must report zero submitted jobs and 12 successful simulations.

## Shared external roots

```bash
export ALIGNN_V26_PACKAGE_ROOT="$HOME/alignn_stage2_v26/delftblue_package_v26"
export ALIGNN_V36_PACKAGE_ROOT="$HOME/alignn_stage2_v36/delftblue_package_v36"
export ALIGNN_GRAPH_CACHE_ROOT="/scratch/$USER/alignn_matbench_is_metal_graph_cache_v26_verified"
export ALIGNN_STAGING_ROOT="$HOME/ml4md/ALIGNN/.stage2_primary_v26_staging"
export ALIGNN_WORK_ROOT="$HOME/ml4md/ALIGNN/stage2_primary_v26"
export ALIGNN_QUARANTINE_ROOT="$HOME/ml4md/ALIGNN/quarantine_v26"
```

## Path A: record the three validation-only diagnostics

```bash
DIAG_JOB=$(sbatch --parsable --export=ALL slurm/11_diagnose_preouter_convergence_v38.sbatch)
DIAG_JOB="${DIAG_JOB%%;*}"
echo "$DIAG_JOB" > "$HOME/v38_convergence_diagnostic_job.txt"
```

This job is expected to complete successfully while leaving the cells scientifically blocked. It must print `V38_PREOUTER_CONVERGENCE_DIAGNOSTIC: RECORDED_AND_BLOCKED`. It never accesses outer-test inputs.

## Path B: recover the five existing exports

```bash
RECOVERY_JOB=$(sbatch --parsable --export=ALL slurm/12_recover_existing_exports_v38.sbatch)
RECOVERY_JOB="${RECOVERY_JOB%%;*}"
echo "$RECOVERY_JOB" > "$HOME/v38_existing_export_recovery_job.txt"
```

Every element must finish `COMPLETED 0:0` and print both `V38_EXISTING_EXPORT_RECOVERY: PASS` and `V38_EXISTING_EXPORT_PROMOTION: PASS`. This path performs no outer-test rematerialization and no metric calculation.

## Final safe status check

```bash
python3 - <<'PY'
from pathlib import Path
root = Path.home() / "ml4md/ALIGNN/stage2_primary_v26"
complete = [(i, *divmod(i, 5)) for i in range(25)
            if (root / f"fold_{i//5}" / f"seed_{i%5}" / "COMPLETE.json").is_file()]
print("complete_count:", len(complete))
print("complete_cells:", complete)
PY
```

Expected after Path B: 22 complete cells. OOF remains blocked until a separately validated MUBen numerical resolution completes indices 6, 14 and 16.
