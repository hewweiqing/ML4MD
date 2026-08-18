# DelftBlue v39 recovery runbook

v39 supersedes v38 operationally only because one inherited runtime test was stale. The v38 recovery implementation and frozen failure sets are unchanged.

## Upload from Windows PowerShell

```powershell
$Base = "C:\Users\User\OneDrive - Delft University of Technology\Master Y1\Q4\Research\experiments\matbench_alignn_is_metal_calibration\stage2"
scp "$Base\alignn_stage2_delftblue_v39.tar.gz" "hhew@login.delftblue.tudelft.nl:~/"
scp "$Base\DELFTBLUE_ARCHIVE_MANIFEST_V39.json" "hhew@login.delftblue.tudelft.nl:~/"
```

## Extract and validate

```bash
cd "$HOME"
echo "$(python3 -c 'import json; print(json.load(open("DELFTBLUE_ARCHIVE_MANIFEST_V39.json"))["archive_sha256"])')  alignn_stage2_delftblue_v39.tar.gz" | sha256sum --check
mkdir -p "$HOME/alignn_stage2_v39"
tar -xzf alignn_stage2_delftblue_v39.tar.gz -C "$HOME/alignn_stage2_v39"
cd "$HOME/alignn_stage2_v39/delftblue_package_v39"
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

After all checks pass, export the roots and submit `slurm/11_diagnose_preouter_convergence_v38.sbatch` and `slurm/12_recover_existing_exports_v38.sbatch` exactly as documented in `V38_RECOVERY_RUNBOOK.md`.
