# DelftBlue v26 regression-test correction

The v25 runtime suite reported `143 passed, 1 failed`. The sole failure was a false-negative source-text assertion. It prohibited the substring `output, elapsed = ...`, which is necessarily contained within the correct identifier text `model_output, elapsed = ...`.

v26 removes only that invalid substring assertion. The separate AST-based regression in `tests/test_v25_profile_output.py` remains and verifies that no store binding named `output` exists while distinct `output_path` and `model_output` bindings do exist. All positive report-write assertions also remain.

The v25 profiler correction itself is unchanged. No dependency, scientific configuration, model, data, split, seed, graph cache logic, Random2 behavior, MUBen scaler, Slurm resource, calibration, OOF, or final-audit calculation changed.

Setup may recover from the preserved, passed v24 login certificate left by v25's fail-closed recertification attempt. It revalidates the unchanged environment and runs the complete v26 tests before writing a new certificate.
