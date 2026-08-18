# ALIGNN Stage-2 v26 experiment protocol

v26 changes only a false-negative test assertion and does not amend the scientific or execution protocol.

Operational release note: additive v25 corrects only a Python local-variable collision in the non-primary 100-batch profiler's report-writing path. It does not amend the frozen scientific protocol. Package-bound gates and explicit profile approval remain mandatory.

v20 preserves the v17 scientific protocol, v18 certification protocol and v19 import-path correction without amendment. Its only operational change is deterministic post-clone restoration of the already-frozen bootstrap-tool versions.

Native logits are exactly `[z_0,z_1]` from `ALIGNN.fc` before `LogSoftmax`. Descriptor extraction is `eval()`/`no_grad()` and must preserve the complete non-head state, including BatchNorm buffers. Metric and clustered-bootstrap definitions are frozen in `CALIBRATION_METRIC_DEFINITIONS.md` and `V17_SCIENTIFIC_INTEGRITY_AMENDMENT.md`. Validation gates cannot require Random2 improvement.

The v16 hash-locked revision-9 dependency environment is retained. v18 changes only when and where certification evidence is created: static resolution is checked before installation; final-prefix runtime evidence and full pytest results are created outside the immutable package and bound into the login certificate. Every GPU script retains the shared activation helper; compute jobs remain uncoupled. The non-primary smoke starts with synthetic execution-contract checks for singleton batches, resume equivalence, OneCycle step count, paired initialization, RNG restoration and temperature invariance.

Gate order remains: package verification, environment setup, static audit, simulation-only `sbatch --test-only`, A100 preflight, graph cache, smoke, 100-batch profile, human approval, primary cell, paired grid, calibration/export, OOF consolidation, and final audit.
