# Coordinate-GPU-v4: Consolidated Project Reference

This is a synthesized, non-redundant reference for the `coordinate_gpu_v4` package,
compiled from all 53 source markdown files (`STATUS.md` plus 52 files under
`package/`). It presents current, final rules (later corrections supersede earlier
statements) plus a condensed changelog. Source files are untouched; this document
is the entry point for understanding where the project stands and how the pipeline
works.

---

## 1. Current Status (as of 2026-08-18)

Recorded from the DelftBlue execution transcript in `STATUS.md`.

**Passed gates:**
- A100 runtime preflight — passed.
- Corrected non-primary smoke test — passed (see v4 permutation-device correction, §5).
- 100-batch A100 profile, job `10657491`: `COMPLETED 0:0`.
  - Profile projection: **4.602648825878898 paired 40-epoch hours**.
  - Peak CUDA memory reserved: **12,922,650,624 bytes** (below the 35 GiB policy cap).
- Profile approval: approved without using scientific outcomes.
- Fold-0/seed-0 primary training, job `10657501`: `COMPLETED 0:0`.
  - Primary training status: `training_complete`, `outer_test_accessed=false`.
- Primary full-grid gate: passed.

**Pending:** the remaining 24 training cells and dependent calibration/export, OOF,
and final-audit jobs were submitted on DelftBlue but **their outputs are not yet in
this GitHub bundle**. Coordinate results should be added only after the final audit
passes and transferred artifacts are hash-verified.

**Active cluster paths:**
```text
package:   $HOME/alignn_coordinate_gpu_v4/delftblue_coordinate_gpu_v4
work root: $HOME/ml4md/ALIGNN/gpu_coordinate_primary_v4
OOF root:  $HOME/ml4md/ALIGNN/gpu_coordinate_oof_v4
authority: $HOME/alignn_stage2_v29/delftblue_package_v29
```
The separate "descriptor-v41" package is **not** a runtime dependency of this
coordinate-v4 experiment.

### How we got to GPU execution
The scientific design was originally CPU-only (packaged as v28/v29, corrected
through v29 for a DGL bug — §5). The certified CPU 100-batch profile projected
**~100.48 hours for one paired 40-epoch cell**; repeated 24-hour CPU Slurm segments
confirmed this was operationally impractical. A prospective device amendment
(2026-08-16, before any coordinate outer-test result was accessed) moved execution
to a single A100 GPU, restarting every cell from its original seed-defined paired
initialization (no CPU state reused). This produced the `coordinate_gpu_v1`–`v4`
lineage; **v4 is the current, passing package** (§5 "GPU device pivot").

---

## 2. Experiment Design (current, frozen scientific protocol)

Source: `RANDOM2_COORDINATE_CPU_PROTOCOL.md`, `EXPERIMENT_PROTOCOL.md`,
`STAGE_2_PROSPECTIVE_AMENDMENT.md`, `GPU_COORDINATE_PROSPECTIVE_DEVICE_AMENDMENT.md`,
`GPU_COORDINATE_PACKAGE_READINESS_REPORT.md`.

### Naming note (avoid confusion)
Two related "Random2" interventions exist in this project's history:
- **Random2-Descriptor** (older; described in `STAGE_2_PROSPECTIVE_AMENDMENT.md` and
  `PROSPECTIVE_ANALYSIS_PLAN.md`): perturbs the pooled crystal-descriptor interface.
- **Random2-Coordinate** (current; this package): perturbs raw Cartesian atomic
  coordinates *before* graph construction.

Both share the same underlying Stage-2 machinery (model, training loop, checkpoint
rule, temperature scaling, metrics, bootstrap). `coordinate_gpu_v4` implements only
the **Coordinate** variant. The Descriptor-variant planning docs are retained here
as design lineage/context, not as the active protocol.

### Task and data
- Benchmark: Matbench `matbench_mp_is_metal`.
- 5 official Matbench outer folds x 5 seeds (0–4) = **25 official cells**.
- Each cell trains **two** CPU/GPU models from one shared initial ALIGNN state:
  **Control** and **Random2-Coordinate**.
- Splits: for every fold/seed, one deterministic stratified validation split is
  formed from outer-training data only (`random_state=seed`). Train/validation/
  outer-test are pairwise disjoint. Runtime authority for splits is
  `ALL_25_SPLIT_HASHES.json` — see caveat in §6.

### Model
- Architecture/training pinned to the official Matbench ALIGNN implementation,
  frozen per the approved continuation prompt. Sources:
  https://github.com/materialsproject/matbench (benchmark `matbench_v0.1_alignn`),
  https://github.com/usnistgov/alignn.
- Historical Matbench leaderboard entry used ALIGNN 2021.12.27, DGL 0.6.1/DGL-CUDA
  0.6.1, PyTorch 1.10.1 — **not reproduced exactly**. This project uses **pinned
  ALIGNN commit `f2366daa3413d28a825b46e34d001b5549b05a40`, ALIGNN package
  2025.4.1, DGL 1.1.1+cu118, PyTorch 2.0.1+cu118** — a current-pinned
  implementation of the historical architecture. The historical result is context
  only, never used for tuning.
- `ALIGNNConfig(classification=True, num_classes=2)` → `fc = Linear(hidden_features, 2)`
  → `LogSoftmax(dim=1)`. **Native raw logits** = the two columns emitted by
  `ALIGNN.fc` immediately before `LogSoftmax`, shape `[batch, 2]`. Every forward
  pass asserts `log_softmax(z)` equals the model's public output and (when labels
  present) `NLLLoss(log_softmax(z), y) == CrossEntropyLoss(z, y)`.
  Pinned trainer loss is `NLLLoss`.

### Supervised training (both branches, identical settings)
- 40 epochs, batch size 32, AdamW, max LR 0.001, weight decay 1e-5, OneCycleLR,
  float32, unweighted NLL. No class weighting, label smoothing, mixed precision,
  gradient accumulation, early stopping, or multi-worker data loading.
- Identical initialization, supervised samples, validation samples, epoch-level
  ordered sample-ID hashes, optimizer, scheduler, batch size, and seeds across
  paired branches.
- Checkpoint selection: minimum inner-validation NLL; **earliest epoch wins ties**.
  No early termination.

### Random2-Coordinate intervention
- Each inner-training structure is deep-copied; independent `N(0, sigma^2)`
  Cartesian displacement is added **per atom and per axis**, without removing the
  mean. Species, cell, and PBC are preserved; only periodic fractional axes are
  wrapped.
- ALIGNN atom and line graphs are **rebuilt from the noisy coordinates** — clean
  cached graphs are never reused for noisy records.
- **Sigma = 0.040 Å per Cartesian axis**, a fixed project-defined dose (not
  data/outcome-selected). Authorization route recorded exactly as
  `project_defined_fixed_dose`.
- Each fold/seed coordinate cache holds exactly **3,000 valid records** in
  deterministic atomic shards (25 cells x 3,000 = 75,000 total). Labels are
  exactly 1,500 zero / 1,500 one, deterministically shuffled, and are **unrelated
  to true targets** (random-label warm-up).
- A common **frozen** CPU/GPU ALIGNN encoder extracts pooled descriptors
  (`eval()`/`no_grad()`). Only `fc.weight` and `fc.bias` receive **938 AdamW
  updates**, batch size 128, learning rate 1e-4, zero weight decay. Encoder
  parameters and BatchNorm buffers (including `num_batches_tracked`) must remain
  **byte-identical** before/after warm-up and between Control/Random2 immediately
  before supervised training.
- Warm-up permutation is generated with a **CPU** `torch.Generator` seeded by
  `seed + 280000` (kept CPU-side even on GPU runs — see §5 GPU v4 correction);
  only the selected index batch is copied to the active CUDA device.
- After warm-up, the supervised Python/NumPy/torch-CPU/CUDA RNGs are **reset to
  seed 0**; fresh AdamW optimizer and OneCycleLR scheduler are created after that
  reset.

### Sigma authorization mechanism
`COORDINATE_NOISE_CONFIG.json` ships with no scientific sigma (fail-closed) until
an externally produced, hash-bound "prospective authority" JSON is imported. That
authority must declare a finite positive `sigma_cartesian_per_axis_angstrom`,
status `approved_prospective_coordinate_sigma`, `scientific_execution_authorized:
true`, `selected_prospectively_without_alignn_outcomes: true`, `test_only: false`,
the package's aggregate and coordinate-template SHA-256, and a 64-character
prospective provenance SHA-256. The runtime gate is
`alignn_stage2.sigma_authorization.require_authorized_sigma()`, called before any
scientific labels, structures, caches, or outer-test artifacts are read; with no
approved artifact it raises `CoordinateSigmaAuthorizationMissing`. The v29
authority actually used has SHA-256 `ea817b08ec5e5d150307e4a26fcbeb561b318c666a33612ee9b2b0182e624fc8`,
origin package aggregate `d91ff9b17b6a75d8b38ecb62efbdbea796560426c365f8bb3a704091606183ef`.

### Reported readouts
Four conditions per cell: **raw** and **MUBen temperature-scaled**, for both
**Control** and **Random2-Coordinate**.

### Primary comparator / estimand (`CROSS_EXPERIMENT_COMPARISON_POLICY.md`,
`PROSPECTIVE_ANALYSIS_PLAN.md`, `V17_SCIENTIFIC_INTEGRITY_AMENDMENT.md`)
- Primary comparison is **paired Random2-Coordinate vs. paired Control within the
  same fold, seed, data order, dependency environment, and execution device**. A
  cross-device comparison (e.g. earlier v26 GPU Control vs. this CPU/GPU
  Random2-Coordinate) is explicitly secondary/descriptive only — never a
  replacement for the paired estimand.
- Estimand: `Delta_s = NLL_D,s^OOF - NLL_B,s^OOF` per seed (5 official outer folds
  concatenated into one OOF vector per seed); point estimate = mean of the 5
  seed-specific deltas. **Negative is favorable.**
- Secondary calibration estimands: Brier, ECE-15, adaptive equal-mass ECE,
  classwise ECE. Performance/failure-awareness endpoints reported separately.
- Bootstrap: **5,000 repetitions**, resampling **structures** (seed `20260715`);
  every sampled structure carries all 5 seeds and all 4 conditions (structure-
  clustered, not fold-independent). Folds are not treated as independent
  replicates. An interval crossing zero is not described as "supported"; no
  equivalence claim without a predeclared margin.
- Validation "adequacy" gates check only basic Control competence and mechanical
  integrity — **no gate compares Random2 to Control or requires Random2 to
  improve on it**. Any protocol change prompted by fold-0/seed-0 evidence requires
  a new package version and restart of all cells.

### Preprocessing policy (`OUTER_TEST_STRUCTURE_PREPROCESSING_POLICY.md`)
One global, label-free graph-preprocessing pass over every structure (train +
outer-test) is permitted before training: it reads structure dictionaries only for
graph conversion, never retains/indexes/hashes labels, and the cache manifest
stores only structure IDs/hashes, graph/runtime/release provenance, shard
locations, offsets, counts, and graph-file hashes. Cache creation must record
`labels_used_in_graph_construction: false`, exactly one graph construction per
structure. Outer-test **labels, logits, predictions, metrics, and outcome
decisions** remain prohibited until a cell passes environment, cache, smoke,
100-batch, training, validation-adequacy, Random2, checkpoint, and final
temperature-approval gates.

---

## 3. Execution Contracts

### CPU execution contract (`CPU_EXECUTION_CONTRACT.md`) — v28/v29 lineage
All v28 Slurm jobs: account `research-ME-mse`, partition `compute`, 1 task, 8
CPUs, `3968M` per CPU. Scientific jobs initially request 24 hours; arrays capped
at 5 concurrent tasks. No GPU requested. Every job exports
`CUDA_VISIBLE_DEVICES=""`; runtime selects `torch.device("cpu")` — model
parameters, graph tensors, logits, losses, and optimizer state stay on CPU. A
runtime preflight rejects a visible accelerator; a static audit rejects GPU
directives, `.cuda()`, autocast, and accelerator-memory ops in v28 scientific
paths. (The certified environment may contain CUDA-enabled PyTorch as inherited
provenance — that does not authorize or cause accelerator execution in v28.)

*Note: this CPU contract governs the original v28/v29 CPU package. The now-active
GPU package (`coordinate_gpu_v4`) instead runs on a single certified A100 — see
`DELFTBLUE_GPU_COORDINATE_RUNBOOK.md` in §4.*

### Deterministic resume contract (`CPU_RESUME_CONTRACT.md`, mirrored for GPU)
Checkpoints atomically preserve model, optimizer, OneCycleLR scheduler, epoch,
next batch position, explicit sample order, Python/NumPy/torch-CPU (and CUDA, for
GPU) RNG state, best validation NLL/checkpoint, history, optimizer-step count, and
package/config/clean-cache/coordinate-cache/authorization hashes. Slurm sends
`SIGUSR1` five minutes before wall time; at the next completed optimizer batch the
process atomically writes `last.pt` + `INCOMPLETE_RESUME_REQUIRED.json` and
**returns exit code 75**. A partial run never writes `COMPLETE.json`. Resubmitting
the same cell with unchanged paths/authority resumes from the exact next batch. A
hash mismatch fails closed — never delete/edit a checkpoint to force continuation.
`COMPLETE.json` is written only after both branches, validation-logit artifacts,
approved validation-only calibration, and prediction export finish.

### Profile approval — non-primary, outcome-neutral (`CPU_PROFILE_APPROVAL_CONTRACT.md`,
`RESOURCE_PROFILE_APPROVAL_CONTRACT.md`)
The 100-batch profile uses fold-0/seed-0 inner-training data only, verifies the
authorized coordinate cache, performs the full coordinate head warm-up, measures
100 full frozen-configuration optimizer batches, and records timing, RSS/CUDA
memory, projected paired 40-epoch time, hashes, and confirmation of no outer-test
access. Profiling **does not** approve primary training — a human must inspect
the profile JSON and run `scripts/review_cpu_profile.py --approve` (CPU) /
`scripts/review_gpu_coordinate_profile.py` (GPU). Approval requires an identified
reviewer, UTC time, decision, and reasons, and binds SHA-256 identities for the
profile, environment certification, package aggregate, execution plan, cache
manifest, resource policy, and primary Slurm script. **Numeric thresholds:**
- Conservative paired upper-bound projection (with declared safety factor) must
  consume **at most 80%** of primary wall time.
- Projected host memory, peak CUDA reserved memory, and projected full-grid disk
  must each consume **at most 80%** of verified capacity.
- Any unavailable, missing, non-finite, contradictory, or changed input
  prohibits approval; a rejected/copied/stale/mismatched approval record blocks
  primary, grid, calibration, OOF, and final-audit jobs. Resource/identity
  changes require a new profile+approval; packaged-file changes require a new
  additive release.

---

## 4. Calibration, Metrics, and Prediction Artifacts

### Temperature scaling (`TEMPERATURE_SCALING_INTERFACE_CONTRACT.md`)
Only approved implementation: `vendor/muben_temperature_scaling.py`, SHA-256
`868654151b87d526f122f741183c23599913541d5d62aada5cb35c91bc510719`, status
`approved_muben_persistent_final_audit_passed`.
```text
fit_temperature(validation_logits, validation_labels) -> fitted object
apply_temperature(logits, fitted object) -> scaled logits
```
One shared scalar `T = exp(log_T)`; fitting minimizes unweighted inner-validation
cross-entropy in **float64** via LBFGS strong-Wolfe against declared
gradient/change tolerances, must report successful numerical convergence. Fitted
record exposes positive finite T, objective, convergence, optimization steps,
before/after validation NLL, implementation version, parameterization, dtype,
tolerances, gradient, and source SHA-256. **Fatal conditions:** fitting on
outer-test inputs, fitting on probabilities instead of logits, vector/per-column
temperatures, pending approval, non-positive T, non-finite values, dimension
changes, raw-array mutation, prediction/ranking changes, source-hash mismatch.
Numerical identity required: `softmax(z/T)[:,1] == sigmoid((z[:,1]-z[:,0])/T)`
within `atol=1e-7, rtol=1e-6`.

### Frozen calibration metric definitions (`CALIBRATION_METRIC_DEFINITIONS.md`, v17)
- **NLL:** mean two-class cross-entropy from native or TS-scaled `[z0,z1]`.
- **ECE-15 (top-label):** confidence = `max(p0,p1)`; correctness = predicted-class
  indicator; 15 equal-width bins over `[0,1]`, `[lower,upper)` except final bin
  includes 1.
- **Adaptive ECE-15:** top-label confidence ECE over 15 deterministic equal-mass
  groups; samples stably sorted by confidence, split via `numpy.array_split`
  (ties retain sample order).
- **Classwise ECE-15:** equal-width ECE computed separately for class 0
  (`p0`, `1[y=0]`) and class 1 (`p1`, `1[y=1]`); reported error = unweighted mean
  of the two.
- **Brier:** binary **positive-class** Brier `mean((p1-y)^2)` — not the two-class
  summed Brier score.
- **ROC-AUC:** from continuous `p1`, average ranks for ties; predicted labels
  never used as the score.
- **Accuracy/F1:** predicted class = `argmax([z0,z1])`; F1 uses class 1 as
  positive.
- **Temperature invariance:** predicted class, margin ordering, tie pattern, and
  ROC-AUC must be unchanged for any positive shared temperature.

### Prediction artifact schema (`PREDICTION_ARTIFACT_SCHEMA.md`)
Every CSV row: `structure_id`, `fold`, `seed`, `condition`, `split`,
`true_label`, `raw_native_logit_0`, `raw_native_logit_1`,
`raw_probability_positive`, `scaled_probability_positive`, `predicted_label`,
`temperature_reference_id`, `checkpoint_sha256`, `package_aggregate_sha256`,
`coordinate_noise_config_sha256`, `sample_order_index`, `execution_device`
(`cpu` in the schema doc — GPU package uses the equivalent GPU device value).

Conditions: `CPU_CONTROL_RAW`, `CPU_CONTROL_TS`, `CPU_RANDOM2_COORDINATE_RAW`,
`CPU_RANDOM2_COORDINATE_TS` (naming carried from the CPU-first schema doc; the
GPU package reuses the same four logical conditions). Splits:
`inner_validation`, `outer_test`. `raw_probability_positive` is **audit-only**,
equal to `softmax([z0,z1])[1] = sigmoid(z1-z0)`; the scaler consumes **logits
only**, never probabilities. `softmax(z/T)[1] = sigmoid((z1-z0)/T)` within
`atol=1e-12` in the exported artifact (tighter than the interface contract's
`1e-7`). Positive scaling preserves predicted class and binary-logit ordering.

### MUBen scaler provenance (`MUBEN_TS_EVIDENCE_AUDIT.md`,
`MUBEN_TS_INTEGRATION_CHECKLIST.md`, `UNIMOL_TS_RECONCILIATION_CHECKLIST.md`)
Status: `approved_muben_persistent_final_audit_passed`. Completed MUBen Uni-Mol
evidence audit: PASS on all 6 expected cells at release commit
`446471d46449ec79cc0f846baa54d7f4a2f90695`; source SHA-256
`f4989a94700990e768cbe595a1399f23dc79e33df2471dd4e12ec32bf5245d43`; evidence
archive SHA-256 `410e96c2257562601646705e4fe6912413bf69d51b9dd9296bbd2c405e4c9c19`.
MUBen's own reproduction used its native optimizer; ALIGNN does not claim that as
its protocol — instead the **approved derived adapter** implements the
prospectively frozen one-task shared-scalar extension (`T=exp(log_T)`, float64
unweighted validation NLL, numerical convergence), pinned at SHA-256
`868654151b87d526f122f741183c23599913541d5d62aada5cb35c91bc510719`. Fully
reconciled: no further action needed unless a new scaler source is adopted via a
new additive package version + new final-audit approval. Pending/generic/
missing/hash-mismatched approval records fail closed before any outer-test
artifact is opened.

---

## 5. Environment, Dependencies, Runbooks

### Certified environment
- Conda prefix: `/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9`
  (`.staging` sibling used during bootstrap).
- CPython **3.10.20**, Linux x86-64, glibc 2.28 (later 2.35), pip **25.3** pinned
  before any production dependency stage.
- Every remote distribution version- and SHA-256-pinned, installed with
  `--no-deps --require-hashes`; full graph checked with `pip check`.
- Bootstrap tool pins (must survive conda clone — see v20 fix): pip 25.3,
  setuptools 80.9.0, wheel 0.45.1.
- CUDA runtime: 11.8 (`DGLBACKEND=pytorch`; `LD_LIBRARY_PATH` includes
  `nvidia/cuda_runtime`, `cublas`, `cusolver`, `cusparse` libs from the env).
- Library versions: PyTorch 2.0.1+cu118, DGL 1.1.1+cu118, ALIGNN 2025.4.1
  (commit `f2366daa3413d28a825b46e34d001b5549b05a40`), Matbench 0.6.
- `importlib-metadata` excluded (markers require Python <3.10); `colorama`
  removed as Windows-only/unreachable; `typing-extensions==4.16.0` selected.
  PyTorch only from its dedicated CUDA-11.8 index; DGL only from its exact
  direct URL — CUDA and ordinary PyPI locks kept disjoint to prevent silent
  cross-resolution.
- Certification artifacts: `DEPENDENCY_RESOLUTION_EVIDENCE.json` (static,
  passed), `CLEAN_INSTALL_EVIDENCE.json`/`CLEAN_INSTALL_TRANSCRIPT.txt`
  (deliberately pending placeholders — only real final-prefix install evidence,
  written under the environment prefix during setup, can certify a runtime).
- Bootstrap install order (v18+): verify static evidence → install/test a
  **staging** prefix in bootstrap mode → clone staging to **final** prefix →
  re-verify final prefix → force-replay `BOOTSTRAP_REQUIREMENTS.txt` with
  `--no-deps --require-hashes` (v20 fix, since conda clone silently upgrades
  bootstrap tools) → write real clean-install evidence/transcript → run full
  pytest (non-bootstrap mode) → write hashed pytest log → issue login
  certificate binding package aggregate + evidence/transcript/pytest-log paths
  and hashes. `require_certification.py` rechecks these bindings; the A100
  report binds the login-certificate hash and package aggregate. Any missing/
  mismatched/tampered artifact fails closed.

### CPU runbook summary (`DELFTBLUE_RUNBOOK.md`, v29 CPU package)
1. `scp` archive + manifest to DelftBlue; extract to `$HOME/alignn_stage2_v29`.
2. `verify_package.py`, `audit_cpu_only.py`, `setup_cpu_environment.sh`,
   `pytest tests_v29`.
3. `delftblue_cpu_test_only.py` — simulation-only resource validation (no jobs
   submitted).
4. `sbatch slurm/00_cpu_preflight.sbatch` — permitted while sigma is pending.
5. Import prospective sigma authority via
   `ALIGNN_COORDINATE_SIGMA_SOURCE` env var →
   `slurm/01_import_verify_coordinate_sigma_cpu.sbatch` → central gate file
   `preflight/VERIFIED_COORDINATE_SIGMA_AUTHORIZATION.json`, checked by
   `alignn_stage2.sigma_authorization.require_authorized_sigma()`.
6. Cache/smoke/profile: `slurm/05_build_coordinate_warmup_cache_cpu.sbatch` →
   `06_smoke_coordinate_fold0_seed0_cpu.sbatch` →
   `07_profile_coordinate_100_batches_cpu.sbatch`, then manual
   `scripts/review_cpu_profile.py --approve` +
   `scripts/require_cpu_profile.py`.
7. Primary + grid: `08_primary_coordinate_fold0_seed0_cpu.sbatch` →
   `10_train_coordinate_fold_seed_cpu.sbatch` (array). Exit 75 = intentional
   incomplete cell; resubmit exact array index (checkpoint reused). Do not
   launch calibration until all 25 cells lack
   `INCOMPLETE_RESUME_REQUIRED.json`.
8. Calibration/OOF/audit: `20_calibrate_export_coordinate_cpu.sbatch` →
   `30_consolidate_coordinate_oof_cpu.sbatch` →
   `40_final_audit_coordinate_cpu.sbatch`.

**This CPU path was abandoned for scientific execution** after the profile
showed ~100.48h/cell (§1); it remains as pilot/operational evidence only.

### GPU runbook (`DELFTBLUE_GPU_COORDINATE_RUNBOOK.md`) — currently active path
Moves only the device from CPU to A100 CUDA; fixed dose, folds, seeds, model,
optimizer, epochs, graph settings, warm-up, scaler, and analysis are unchanged.
Partial CPU checkpoints are retained as evidence but **never imported**.
1. `scp` archive `alignn_coordinate_gpu_delftblue_v4.tar.gz` + manifest.
2. Extract to `$HOME/alignn_coordinate_gpu_v4`; verify archive SHA-256 against
   manifest; `cd delftblue_coordinate_gpu_v4`.
3. Export env: `ALIGNN_ENV_PREFIX`, `ALIGNN_CLEAN_INSTALL_EVIDENCE`,
   `ALIGNN_CLEAN_INSTALL_TRANSCRIPT`, `DGLBACKEND=pytorch`, `LD_LIBRARY_PATH`
   (nvidia cuda_runtime/cublas/cusolver/cusparse libs).
4. `scripts/verify_package.py`, `pytest tests_gpu`,
   `scripts/audit_gpu_coordinate_resources.py`,
   `scripts/delftblue_test_only.py` (`sbatch --test-only` simulation only).
5. Define isolated roots: `ALIGNN_DATASET`, `ALIGNN_GRAPH_CACHE_ROOT`
   (`..._graph_cache_v26_verified`), `ALIGNN_COORDINATE_CACHE_ROOT`
   (`..._coordinate_cache_v29`), `ALIGNN_V29_ROOT`,
   `ALIGNN_COORDINATE_SIGMA_AUTHORIZATION` (points at the v29 authority file —
   byte-identical, verified by v2's origin-lineage check),
   `ALIGNN_GPU_COORDINATE_WORK_ROOT`, `ALIGNN_GPU_COORDINATE_OOF_ROOT`. Package
   refuses `ALIGNN_GPU_COORDINATE_WORK_ROOT` set to the old CPU root.
6. Gates, one job at a time, each checked for `COMPLETED 0:0` + a named PASS
   line before continuing: `00_gpu_coordinate_a100_preflight.sbatch` →
   `06_smoke_coordinate_f0s0_gpu.sbatch` →
   `07_profile_coordinate_100_batches_gpu.sbatch` → manual
   `scripts/review_gpu_coordinate_profile.py --decision approve`.
7. Primary: `08_primary_coordinate_f0s0_gpu.sbatch` (exit 75 = resumable,
   resubmit identical command) → `scripts/write_gpu_coordinate_primary_gate.py`.
8. Remaining grid (only after primary gate exists):
   `10_train_coordinate_grid_gpu.sbatch` → (afterok)
   `20_calibrate_export_coordinate_gpu.sbatch` → (afterok)
   `30_consolidate_coordinate_oof_gpu.sbatch` → (afterok)
   `40_final_audit_coordinate_gpu.sbatch`. A training-array exit-75 leaves its
   `afterok` dependency unsatisfied; resubmit only that array index, verify all
   25 `TRAINING_STATUS.json` files, then submit a **fresh** downstream chain —
   never release downstream jobs from partial cells.
9. Monitoring: `squeue -u "$USER"`; `sacct -X -j "$JOB_ID" --format=...`.
10. Note: the coordinate-GPU workflow shares the A100 allocation with an
    existing descriptor-GPU array — this affects queueing only, not scientific
    independence.

### Gate order (fixed, `EXPERIMENT_PROTOCOL.md`)
package verification → environment setup → static audit → simulated
`sbatch --test-only` → A100 preflight → graph cache → smoke →
100-batch profile → human approval → primary cell → paired grid →
calibration/export → OOF consolidation → final audit.

### Packaging audit (`AUDIT_REPORT.md`)
Finalized by a deterministic builder: Python AST compilation, JSON parsing,
shell syntax, CPU-only static policy, focused tests, package-manifest
verification, archive path/duplicate/permission safety, clean extraction, and
repetition of the same checks post-extraction. Preservation checks bind
immutable v26 archive SHA-256
`f4781d8946145202f22633a6631283335f7fb012490a472002335927ea5f8996`, blocked v27
archive SHA-256 `cdcfb080fa31236d4d5be3bc23441dcb3a21c1851334f306d6f5bac54988c6f4`,
and the MUBen module SHA-256. Scientific training/inference/calibration/Slurm/
pip/network/outer-test access are prohibited during packaging.

---

## 6. Version History / Changelog (condensed)

Each package version is additive over its predecessor and immutable once
released. Only versions with lasting relevance to understanding the current
pipeline get more than one line.

| Version | What changed |
|---|---|
| v5 | Archive manifest hash mismatch traced to **OS-dependent path sorting** (Windows vs. Linux lexical order) in the aggregate-hash generator; also found stray `.pyc`/`__pycache__` debris (didn't affect the hash but violated hygiene). |
| v6 | Fix for v5: sorts canonical POSIX relative path strings lexically before hashing; independent reconstruction verified. (Not separately documented, implied by v5 doc.) |
| v7 | Test report: 29/29 passed in package + clean extraction. Added installed-source provenance, vendored-source determinism, vector-temperature rejection, extreme-logit, and fold/seed isolation tests. |
| v10 | 47/47 passed. Added final-status/hash approval, positive-`log_T` checks, all-25-split-hash mapping, staged 100-batch/primary gates, structure-shard boundary checks, atomic checkpoint promotion, hash-verified resume. Also fixed the **split-hash reconciliation** issue (see below). |
| v11 | 62/62 passed. Added cache provenance, stale-input rejection, atom/line/metadata corruption quarantine+rebuild, interrupted-shard recovery, host/GPU/disk requirement checks, profile-approval bindings. |
| v12 | 75/75 passed. Handled DelftBlue's system Python 3.6.8 incompatibility via an isolated Python 3.10.12 module subshell; added Slurm `gpu-a100` headers and `8000M` job configs. |
| v13 | 89/89 passed. Slurm resource-header hardening: partition memory-unit parsing, non-exclusive-node rejection, mocked test-only success/failure paths. |
| v14 | 100/100 passed. Fixed CUDA runtime issues: missing cuSPARSE / search-path failures, exact CUDA component pins, shared CUDA-activation helper sourced by all 7 GPU scripts, compute-script decoupling. |
| v15 | 108/108 passed. Real **online dependency resolution**: exact cu118 torch wheel, exact DGL wheel URL, 79-distribution/83-wheel Linux CPython-3.10 graph, 74-wheel lock replay with `--require-hashes`. Uncovered that `exceptiongroup`/`tomli` were unpinned under the pytest marker graph. |
| v16 | **Environment setup correction.** Fixed the v15 gap: pins CPython 3.10.20, pip 25.3 before install, hashes every remote distribution, installs `--no-deps --require-hashes`, checks with `pip check`. This pin set is still current. |
| v17 | **Major: scientific-integrity amendment + implementation.** Froze raw-logit provenance (`ALIGNN.fc` output pre-`LogSoftmax`, with runtime equivalence assertions), singleton-batch normalization, complete encoder/BatchNorm-buffer immutability checks around Random2 warm-up, `CALIBRATION_METRIC_DEFINITIONS.md` (ECE variants), structure-clustered bootstrap, and explicitly **outcome-neutral validation gates** (no requirement that Random2 beat Control). These definitions are still authoritative (§4). |
| v18 | **Bootstrap-certification amendment.** Fixed a circular certification gate: static resolution evidence alone could not certify a runtime. Introduced the staging→clone→final-prefix bootstrap sequence with real post-install evidence, still current (§5). |
| v19 | **Import-path correction.** Real DelftBlue v18 run built/installed successfully but failed at final-prefix creation because `verify_environment.py` didn't add the package root to `sys.path`. Fixed by explicit `PYTHONPATH` prefixing in setup + `__file__`-derived path insertion in the verifier. |
| v20 | **Conda-clone overlay correction.** Real v19 run (138 tests passed) found that Conda's post-clone package relinking silently upgraded pinned bootstrap tools (pip/setuptools/wheel). Fixed by force-replaying `BOOTSTRAP_REQUIREMENTS.txt` with `--no-deps --require-hashes` after the clone, before final verification — still current (§5). |
| v25 | **Profiler output correction.** Real DelftBlue profile job `10633397` crashed with `AttributeError: 'Tensor' object has no attribute 'parent'` — a variable-name collision (`output` reused for both the report path and the model tensor) in the 100-batch profiler. Fixed by renaming to `output_path`/`model_output`. |
| v26 | **Test correction.** v25's regression suite had a false-negative assertion (banned substring `output, elapsed = ...` also matched the correct `model_output, elapsed = ...`); removed the invalid assertion, kept the real AST-based regression check. |
| v28 | Base CPU-only Random2-Coordinate package (implementation-complete; scientific execution deliberately unauthorized pending sigma import). Contains a DGL 1.1.1 defect fixed in v29. |
| v29 | **DGL column correction.** Real coordinate-cache array job `10638930` rejected every record because DGL 1.1.1's `Frame.values()` returns lazy `Column` objects incompatible with `torch.isfinite`. Fixed by materializing features by sorted key before validation. Produced the working 75,000-record CPU coordinate cache reused by the GPU package. Also the origin of `README.md`/`RANDOM2_COORDINATE_CPU_PROTOCOL.md` as currently read. |
| GPU device pivot (2026-08-16) | **Major pivot.** CPU 100-batch profile projected ~100.48h/paired-cell — operationally impractical. Amendment moves execution to a single certified A100; every cell restarts from its original seed-defined initialization (no CPU state reused); same dose/folds/seeds/model/optimizer/scheduler/warm-up/scaler. |
| GPU coordinate v1 | Initial CPU→GPU port; correctly required the v29 sigma authorization. |
| GPU coordinate v2 | **Authorization correction.** v1's validator compared the immutable v29 authorization's origin-package hash against the wrong (GPU execution-wrapper) aggregate, causing false-closed rejection. Fixed to accept the byte-identical v29 authorization. |
| GPU coordinate v3 | **Runtime-helper correction.** v2 omitted `scripts/verify_cuda_runtime.py`, needed by the CUDA-activation script every GPU job sources; DelftBlue preflight job `10653898` failed before any scientific execution. Restored the file from the v26 certified lineage. |
| GPU coordinate v4 (current) | **Permutation-device correction.** v3 smoke job `10654092` passed all resource gates but crashed building the coordinate warm-up permutation: `RuntimeError: Expected a 'cpu' device type for generator but found 'cuda'`. Fixed by keeping the warm-up permutation generator CPU-side (seed `seed+280000`) and copying only the selected index batch to CUDA. **This is the package behind the current passing gates in §1.** |

### Split-hash reconciliation (still-relevant caveat, from v10 / `SPLIT_HASH_RECONCILIATION.md`)
`ALL_25_SPLIT_HASHES.json` (generated with `random_state=seed`) is the
**authoritative runtime split file** — it freezes all 25 `(fold, seed)` cells with
ID/coverage/overlap hashes and outer-test ID hashes (no labels stored). The older
`STAGE_0B_AUDIT.json` was generated with `random_state=fold` despite prose calling
it "seed-0," so its 5 records only match the diagonal cells `(0,0)…(4,4)` — using
it as a general seed-0 check would incorrectly reject folds 1–4/seed 0 and leave
20 cells unchecked. It is preserved as historical evidence but is **not** the
runtime authority. Every training invocation verifies train IDs, validation IDs,
and validation labels against `ALL_25_SPLIT_HASHES.json` before graph
loading/optimization.

---

## 7. Open Issues, Caveats, and Known Limitations (as of latest files)

1. **Results not yet transferred.** Per `STATUS.md`, 24 of 25 training cells and
   all downstream calibration/OOF/final-audit jobs are pending on DelftBlue; no
   scientific numbers exist in this GitHub bundle yet. Do not treat any file here
   as containing outer-test results.
2. **CPU path is dead weight for scientific execution** — retained only as
   pilot/operational evidence (the fold-0/seed-0 CPU run that hit the ~100h/cell
   wall). Do not resume it or import its checkpoints into the GPU work root; the
   package actively refuses a work root pointed at the CPU path.
3. **A100 contention.** The coordinate-GPU workflow shares the A100 allocation
   with an existing "descriptor" GPU array; this can affect queue latency (not
   correctness).
4. **Naming carryover.** `PREDICTION_ARTIFACT_SCHEMA.md` still uses `CPU_`-
   prefixed condition strings (`CPU_CONTROL_RAW`, etc.) and `execution_device:
   cpu`; the GPU package presumably substitutes the equivalent GPU device value,
   but no GPU-specific schema doc explicitly restates this — verify condition
   naming against actual exported CSVs once cells complete.
5. **Fail-closed by design in several places** — treat any of the following as a
   hard stop, not a bug to work around: sigma authorization missing/mismatched,
   profile approval missing/stale, checkpoint hash mismatch on resume, MUBen
   source hash mismatch, split-hash mismatch against `ALL_25_SPLIT_HASHES.json`.
6. **Two "Random2" variants exist in the docs** (Descriptor vs. Coordinate — §2).
   `PROSPECTIVE_ANALYSIS_PLAN.md` and `STAGE_2_PROSPECTIVE_AMENDMENT.md` describe
   the Descriptor variant's shared machinery/estimand language; when reading
   them, mentally substitute "Random2-Coordinate" per this package's actual
   intervention.
7. **No equivalence claims without a predeclared margin**, and an OOF interval
   crossing zero must not be reported as "supporting" a conclusion — this is a
   standing analysis-policy constraint for whoever writes up results, not just a
   historical note.
8. Immutable/blocked-version bookkeeping still active: v26 archive SHA-256
   `f4781d8946145202f22633a6631283335f7fb012490a472002335927ea5f8996` and blocked
   v27 archive SHA-256 `cdcfb080fa31236d4d5be3bc23441dcb3a21c1851334f306d6f5bac54988c6f4`
   remain the preserved-lineage reference points that every later package
   verifies against.

---

## 8. Quick-Reference Facts

- **Dataset:** Matbench `matbench_mp_is_metal`; 5 outer folds x 5 seeds = 25 cells.
- **Dose:** 0.040 Å per Cartesian axis, fixed (`project_defined_fixed_dose`).
- **Coordinate cache:** 3,000 records/cell, 75,000 total; labels 1,500/1,500,
  deterministically shuffled, unrelated to true targets.
- **Warm-up:** 938 steps, batch 128, AdamW lr 1e-4, wd 0; encoder frozen; only
  `fc.weight`/`fc.bias` trained; permutation seed offset `+280000`, CPU generator.
- **Supervised training:** 40 epochs, batch 32, AdamW max-lr 0.001, wd 1e-5,
  OneCycleLR, float32, unweighted NLL; checkpoint = min validation NLL (earliest
  epoch on tie).
- **Temperature scaling:** shared scalar `T=exp(log_T)`, LBFGS strong-Wolfe,
  float64, `vendor/muben_temperature_scaling.py` SHA-256
  `868654151b87d526f122f741183c23599913541d5d62aada5cb35c91bc510719`.
- **Bootstrap:** 5,000 reps, structure resampling, seed `20260715`.
- **CPU resource profile (abandoned):** account `research-ME-mse`, partition
  `compute`, 1 task, 8 CPUs, 3968M/CPU, 24h initial request, array cap 5;
  projected ~100.48h per paired 40-epoch cell.
- **GPU resource profile (active, job `10657491`):** single A100; projected
  4.602648825878898 paired 40-epoch hours; peak CUDA reserved 12,922,650,624
  bytes (policy cap 35 GiB); approval requires all projections ≤80% of verified
  capacity/wall time.
- **Environment:** conda prefix
  `/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9`; CPython 3.10.20;
  pip 25.3; PyTorch 2.0.1+cu118; DGL 1.1.1+cu118; ALIGNN 2025.4.1 (commit
  `f2366daa3413d28a825b46e34d001b5549b05a40`); CUDA 11.8.
- **Cluster paths (GPU v4):**
  - package: `$HOME/alignn_coordinate_gpu_v4/delftblue_coordinate_gpu_v4`
  - work root: `$HOME/ml4md/ALIGNN/gpu_coordinate_primary_v4`
  - OOF root: `$HOME/ml4md/ALIGNN/gpu_coordinate_oof_v4`
  - sigma authority: `$HOME/alignn_stage2_v29/delftblue_package_v29`
- **Prediction CSV fields:** `structure_id, fold, seed, condition, split,
  true_label, raw_native_logit_0, raw_native_logit_1, raw_probability_positive,
  scaled_probability_positive, predicted_label, temperature_reference_id,
  checkpoint_sha256, package_aggregate_sha256, coordinate_noise_config_sha256,
  sample_order_index, execution_device`.
- **Gate order:** package verification → environment setup → static audit →
  `sbatch --test-only` → A100 preflight → graph cache → smoke → 100-batch
  profile → human approval → primary cell → paired grid → calibration/export →
  OOF consolidation → final audit.
