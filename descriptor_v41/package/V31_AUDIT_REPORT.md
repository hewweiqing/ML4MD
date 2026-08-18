# v31 additive recovery audit

v31 is an additive calibration-recovery release derived from immutable v26. It preserves the v26 archive, frozen model configuration, official folds, deterministic inner splits, seeds, Control/Random2 pairing, training optimizer, scheduler, epoch count, graph settings, dataset hash and checkpoint artifacts.

The only scientific numerical amendment is documented in `V31_NUMERICAL_CONVERGENCE_AMENDMENT.md`. The MUBen-derived scaler remains one shared positive scalar for one binary task and consumes native `[n,2]` logits. The v31 approval record is exact-hash fail-closed.

The v26 structure cache is reused only under its pinned aggregate identity `1c15f877cee4eb6eebfd096962c6a793d0d3f14385287a6a9385e3adce05bd7d`; v31 does not falsely relabel that cache. The recovery helper verifies the immutable v26 package, old MUBen source, trained cell inputs, Control/Random2 alignment and absent outer-test sentinel before validation-only fitting.

Packaging performs no training, calibration, inference, pip, network, Slurm or cluster operation and accesses no outer-test artifact. Source and clean-extraction checks cover Python parsing, JSON validation, shell syntax, manifest hashes, archive paths, permissions, Slurm resource policy and v31 behavioral assertions.
