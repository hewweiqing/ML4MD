# v19 package-import correction

Real DelftBlue v18 setup successfully built and installed ALIGNN 2025.4.1 and Matbench 0.6, activated the pinned CUDA runtime, and passed `pip check`. It then failed before final-prefix creation because the staging invocation of `scripts/verify_environment.py` did not place the immutable package root on `sys.path`.

v19 corrects that exact operational defect in two complementary ways:

- every setup invocation of `verify_environment.py` explicitly prefixes `PYTHONPATH` with `$PACKAGE_DIR`;
- `verify_environment.py` derives its package parent from `__file__` and inserts it before importing `alignn_stage2`.

The Slurm entry points were audited and already export the submit directory on `PYTHONPATH`; no Slurm directive or scientific command was changed. The dependency locks, installed-package design, environment prefix, CUDA runtime, model, data, splits, seeds, training, calibration and analysis remain unchanged.

The failed v18 diagnostic is valid operational evidence. It contains no scientific output and no scientific run was affected.
