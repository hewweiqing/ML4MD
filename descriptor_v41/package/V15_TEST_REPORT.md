# v15 local test report

Source-tree suite: **108 passed**.

Release-time online resolution evidence: exact torch cu118 wheel downloaded from the PyTorch cu118 index; exact DGL wheel downloaded from its fixed URL; corrected four-wheel CUDA runtime downloaded from PyPI; combined Linux CPython-3.10 scientific graph resolved to 79 distributions/83 wheel files; full 74-wheel Python lock replay passed with `--require-hashes` and the exact local torch wheel. Filenames, versions, declared dependencies, sizes and SHA-256 values are recorded in `DEPENDENCY_RESOLUTION_EVIDENCE.json`.

Regression coverage includes the nonexistent v14 nvJitLink pin, missing locked distributions, hash mismatch, missing required SONAME/search path, loader failure, stale/partial-prefix rejection, fail-marked Conda clone/certification ordering, v5/v7 certification rejection, shared helper usage in all seven GPU jobs, compute-only decoupling, Slurm resources, fake test-only validation, and absence of bundled profile approval.

No training, calibration, inference, Slurm, cluster operation or outer-test evaluation was performed.
