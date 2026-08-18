# Prospective CPU-to-A100 device amendment

This amendment is recorded on 2026-08-16 before any Random2-Coordinate
outer-test logits, predictions, metrics, calibration results, or comparisons
have been accessed. The device decision uses runtime feasibility only: the
validated CPU 100-batch profile projected approximately 100.48 hours for one
paired 40-epoch cell, and repeated 24-hour CPU segments confirmed that this is
operationally impractical.

The fixed project-defined dose remains 0.040 angstrom per Cartesian axis, with
the authorization route recorded exactly as `project_defined_fixed_dose`. The
official folds, five seeds, deterministic inner splits, cached perturbed
structures, random labels, 938-step coordinate warm-up, model, graph settings,
optimizer, scheduler, epochs, batch size, checkpoint rule, native-logit
contract, validated MUBen temperature scaler and OOF analysis are unchanged.

The incomplete CPU fold-0/seed-0 run is retained as non-primary runtime/pilot
evidence. No CPU model state, optimizer state, selected checkpoint, validation
logit, or partial history may enter the GPU experiment. Every GPU cell,
including fold 0/seed 0, starts from its original seed-defined paired model
initialization in a new work root. All paired branches and all 25 cells use the
same A100 execution path.

The GPU workflow remains blocked until its own A100 preflight, non-primary
smoke, deterministic resume contract, 100-batch profile, manual resource
approval and fold-0/seed-0 primary gate pass. A failed gate stops execution;
another device or dose must not be substituted after observing outcomes.
