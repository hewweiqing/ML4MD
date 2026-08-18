# v19 verification report

The regression suite executes `scripts/verify_environment.py --help` from outside the package with `PYTHONPATH` removed, checks all three setup verifier calls for an explicit package root, and checks every Slurm entry point for submit-directory `PYTHONPATH` export. Source and clean-extraction package integrity, Python parsing, JSON validation, bash syntax, dependency resolution, resource policy, archive safety and permissions are also required before release.

No pip, network, Slurm, training, calibration, inference, graph-cache or outer-test operation is performed during v19 construction.
