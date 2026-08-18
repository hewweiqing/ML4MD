# GPU Coordinate Package Readiness Report

## Scope

This is an additive execution-device amendment for the coordinate Random2 experiment. The prior v29 and v30 artifacts remain immutable. The failed/timed-out CPU runs are retained as operational evidence, not scientific cells.

## Preserved scientific design

- Matbench `matbench_mp_is_metal`, five official outer folds and seeds 0–4.
- Two paired models per fold/seed: Control and Random2-Coordinate.
- Four reported readouts: raw and MUBen temperature-scaled for both models.
- Project-defined fixed dose of 0.040 Å independently per Cartesian axis.
- Frozen graph construction, ALIGNN architecture, 40 epochs, optimizer, scheduler, inner validation, coordinate warm-up, and outer-test policy.
- Approved MUBen source file SHA-256 `108b3183400725ca53e025c6f0f92690f35069f6fd786091b9d37dc50e90f08b`.

## Operational change

Every coordinate cell is restarted from its original seed-specific paired initialization on one A100. No CPU checkpoint or CPU optimizer state is accepted. CUDA RNG state, CPU RNG state, NumPy/Python RNG state, optimizer, scheduler, exact sample order, epoch and batch position are checkpointed for deterministic same-runtime resume.

GPU-coordinate v3 job `10654092` passed its resource gates but stopped before its non-primary smoke completed because a CUDA generator was passed to a CPU `torch.randperm`. Additive v4 preserves the frozen CPU permutation stream and transfers only the selected index batch to CUDA. No dose, label, split, model, optimizer, or outcome rule changed.

## Fail-closed gates

Scientific training remains blocked until the package-local A100 preflight, synthetic/non-primary smoke, 100-batch full-configuration profile, explicit resource approval, and fold-0/seed-0 primary training gate pass. Calibration/export remains blocked until training completes. Outer-test access begins only after both inner-validation temperature fits and Control adequacy pass.

## Reused immutable external evidence

- Certified CUDA 11.8 environment.
- v26 verified label-free clean structure cache.
- v29 exhaustive 75,000-record coordinate cache.
- v29 external fixed-sigma authorization.

These are data/cache/authorization inputs only. Model state is not reused.

## Pending DelftBlue evidence

- GPU-coordinate A100 preflight.
- GPU-coordinate v4 smoke (v3 job `10654092` is retained as failed operational evidence).
- GPU-coordinate 100-batch runtime/memory profile and human approval.
- Primary and remaining scientific GPU cells.
- Calibration/export, OOF consolidation and final audit.

No job was submitted, no training/calibration/inference was run, and no outer-test result was inspected while building this archive.
