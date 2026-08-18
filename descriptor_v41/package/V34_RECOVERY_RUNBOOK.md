# DelftBlue v34 calibration-recovery runbook

v34 is operationally identical to v33 and corrects only one inherited test count. Reuse the certified environment and completed v26 fold-0/seed-0 artifacts.

```bash
cd "$HOME/alignn_stage2_v34/delftblue_package_v34"
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

Only after package verification, all 152 tests and all ten Slurm simulations pass should `slurm/09_recover_calibration_f0s0.sbatch` be submitted using the v33/v32 documented v26 roots.
