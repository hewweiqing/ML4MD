# v29 Test Report

Packaging-workstation checks: 54 dependency-light behavioral/static assertions passed; CPU-only audit passed for ten Slurm jobs and six scientific modules; all Python files parsed; all JSON files parsed; all shell/Slurm files passed `bash -n`; package manifest and clean-extraction manifest matched; archive safety and normalized permissions passed.

The workstation does not contain pytest, DGL, ALIGNN, torch or the frozen scientific dependency environment. `tests_v29/test_delftblue_runtime_pending.py` is therefore retained for the certified DelftBlue Python prefix and is explicitly pending there. It covers real CPU ALIGNN placement, head-only warm-up/encoder immutability, batch-size-one native logits, rebuilding noisy atom/line graphs, and DGL 1.1.1 lazy-Column materialization.

No pip, network, Slurm, training, calibration, inference or outer-test evaluation was run.
