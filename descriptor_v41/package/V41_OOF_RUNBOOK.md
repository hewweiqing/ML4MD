# DelftBlue v41 OOF and final-audit runbook

Use this package only after v40 tasks 6, 14 and 16 complete and `scripts/audit_grid_complete.py` reports `ALIGNN_25_CELL_GRID: PASS`. It does not train, recalibrate, or repeat outer-test export.

## Upload and verify

```powershell
$Base = "C:\Users\User\OneDrive - Delft University of Technology\Master Y1\Q4\Research\experiments\matbench_alignn_is_metal_calibration\stage2"
scp "$Base\alignn_stage2_delftblue_v41.tar.gz" "hhew@login.delftblue.tudelft.nl:~/"
scp "$Base\DELFTBLUE_ARCHIVE_MANIFEST_V41.json" "hhew@login.delftblue.tudelft.nl:~/"
```

```bash
cd "$HOME"
echo "$(python3 -c 'import json; print(json.load(open("DELFTBLUE_ARCHIVE_MANIFEST_V41.json"))["archive_sha256"])')  alignn_stage2_delftblue_v41.tar.gz" | sha256sum --check
mkdir -p "$HOME/alignn_stage2_v41"
tar -xzf alignn_stage2_delftblue_v41.tar.gz -C "$HOME/alignn_stage2_v41"
cd "$HOME/alignn_stage2_v41/delftblue_package_v41"
mkdir -p logs
```

## Bind immutable evidence and completed outputs

```bash
export ALIGNN_ENV_PREFIX="/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9"
export ALIGNN_V26_PACKAGE_ROOT="$HOME/alignn_stage2_v26/delftblue_package_v26"
export ALIGNN_V36_PACKAGE_ROOT="$HOME/alignn_stage2_v36/delftblue_package_v36"
export ALIGNN_GRAPH_CACHE_ROOT="/scratch/$USER/alignn_matbench_is_metal_graph_cache_v26_verified"
export ALIGNN_WORK_ROOT="$HOME/ml4md/ALIGNN/stage2_primary_v26"
export ALIGNN_OOF_OUTPUT="$ALIGNN_WORK_ROOT/oof_analysis_v41"
```

## Submit consolidation and dependent final audit

```bash
OOF_SUBMISSION=$(sbatch --parsable --export=ALL slurm/30_consolidate_oof.sbatch)
OOF_JOB="${OOF_SUBMISSION%%;*}"
test -n "$OOF_JOB" || { echo "STOP: no OOF job ID returned"; exit 1; }
printf '%s\n' "$OOF_JOB" > "$HOME/v41_oof_job.txt"

AUDIT_SUBMISSION=$(sbatch --parsable --dependency=afterok:$OOF_JOB --export=ALL slurm/40_final_audit.sbatch)
AUDIT_JOB="${AUDIT_SUBMISSION%%;*}"
test -n "$AUDIT_JOB" || { echo "STOP: no final-audit job ID returned"; exit 1; }
printf '%s\n' "$AUDIT_JOB" > "$HOME/v41_final_audit_job.txt"

echo "OOF_JOB=$OOF_JOB"
echo "AUDIT_JOB=$AUDIT_JOB"
```

Require `ALIGNN_OOF_CONSOLIDATION: PASS`, followed by `ALIGNN_PERSISTENT_FINAL_AUDIT: PASS`. The final audit remains dependency-blocked if consolidation fails.
