# Descriptor-v41 Consolidated Project Reference

This document consolidates all 62 markdown files under `descriptor_v41/` (and
`descriptor_v41/package/`) into one current-state reference. It supersedes
none of the source files — those remain the append-only, hash-audited record
— but this file is the place to start. Per-file details for the long
V5–V40 correction chain are condensed in §13; the actual current rules (data
splits, resource profiles, calibration metrics, schemas, contracts) are given
in full in §2–§11, using the latest/most-corrected version of each.

## 1. Current Status (as of v41, 2026-08-18)

The pipeline is **complete**. Per `ARTIFACTS.md` and `V41_OOF_RUNBOOK.md`:

- `results/FINAL_AUDIT.json` records **25 valid (fold, seed) cells, 50 trained
  branches (Control + Random2 per cell), 100 readouts (A/B/C/D conditions ×
  25 cells), and no failures**.
- `results/OOF_ANALYSIS.json` contains the completed out-of-fold analysis and
  clustered bootstrap outputs.
- The full 25-cell grid only reached completion after the v40 numerical
  resolution of its last 3 blocked cells (see §13); v41 made no scientific or
  computational change — it only repointed the OOF-consolidation and
  final-audit jobs at the correct, already-existing evidence (see §1.1).
- Large row-level artifacts are **excluded from git**:

  | Artifact | Bytes | SHA-256 |
  |---|---:|---|
  | `oof_predictions.csv` | 419,627,926 | `65cc032c583d18bfcce6c45a85d36ae5343c5cff622e4b1e7588489a35d82ab7` |
  | `alignn_descriptor_v41_final_results.tar.gz` | 82,247,032 | `24bad294760897634cfabbf13734fd303bef0d970844591ce22299d4f798daf5` |

  Their source locations in the research workspace:
  ```text
  imports/alignn_descriptor_v41_final/extracted/oof_analysis_v41/oof_predictions.csv
  imports/alignn_descriptor_v41_final/alignn_descriptor_v41_final_results.tar.gz
  ```
  Do not recompress or edit either while keeping its recorded hash; publish
  separately if row-level reproducibility is required.

**Important reader caveat:** none of the 62 source documents state the
scientific *conclusion* (whether Random2-Descriptor pretraining actually
improved or hurt calibration/NLL relative to Control). They document that the
pipeline ran correctly, end-to-end, with all integrity gates passing. The
answer to the prospective research question (§2.1) lives in
`OOF_ANALYSIS.json` / the final results archive, not in these markdown files.

### 1.1 v41's own change (most recent operational fact)

After DelftBlue v40 completed the final 3 authorized numerical-resolution
exports and the full 25-cell hash audit passed, the *inherited* OOF and
final-audit Slurm jobs still pointed at package-local A100/profile evidence
that does not exist in a clean, additive recovery package. v41 changes only
the operational evidence *paths* for OOF consolidation and final audit: both
jobs now require the existing v26 certification/profile/cache evidence, the
v36 primary gate, the v37 external-input validator, and a fresh 25-cell hash
audit before opening any prediction artifact. The v40 consolidation and
final-audit calculations, five seeds, 5,000 structure-cluster bootstrap
repetitions, MUBen implementation, predictions, and scientific configuration
are all unchanged. No training/calibration/inference/OOF/outer-test/network/
pip/Slurm/cluster operation was performed to produce this correction.

## 2. Scientific Protocol (frozen, current)

Primary sources: `EXPERIMENT_PROTOCOL.md`, `STAGE_2_PROSPECTIVE_AMENDMENT.md`,
`PROSPECTIVE_ANALYSIS_PLAN.md`, `V17_SCIENTIFIC_INTEGRITY_AMENDMENT.md`.

### 2.1 Question and estimand

Does Random2-Descriptor pretraining at the pooled crystal-descriptor
interface add calibration or failure-awareness benefit beyond
validation-fitted temperature scaling in native ALIGNN classification of
Matbench `matbench_mp_is_metal`? Random2-Descriptor is described as "a
latent-interface adaptation of random-label warm-up for a periodic crystal
classifier; it is not raw-coordinate noise training."

Primary estimand: paired structure-level five-fold OOF difference in NLL,
`Delta_s = NLL_D,s^OOF - NLL_B,s^OOF` (condition D = Random2+TS minus
condition B = Control+TS), computed per seed `s`. Negative is favorable. The
point estimate is the mean of the five seed-specific deltas. Secondary
calibration estimands: Brier score, ECE-15, adaptive equal-mass ECE,
classwise calibration error. Performance and failure-awareness endpoints are
reported separately.

### 2.2 Model, provenance, versions

Architecture and training settings are taken from the official Matbench
ALIGNN implementation, frozen per the approved continuation prompt:

- https://github.com/materialsproject/matbench/tree/main/benchmarks/matbench_v0.1_alignn
- https://matbench.materialsproject.org/Full%20Benchmark%20Data/matbench_v0.1_alignn/
- https://github.com/usnistgov/alignn

The historical Matbench submission used ALIGNN 2021.12.27, DGL 0.6.1, PyTorch
1.10.1. This experiment instead pins **ALIGNN commit
`f2366daa3413d28a825b46e34d001b5549b05a40`, ALIGNN package 2025.4.1, DGL
1.1.1+cu118, PyTorch 2.0.1+cu118** — a current-pinned implementation of the
historical architecture, not an exact reproduction. The historical leaderboard
result is context only, never used for tuning.

`ALIGNNConfig(classification=True, num_classes=2)` constructs the native
`fc = Linear(hidden_features, 2)` and `LogSoftmax(dim=1)`. In `forward`, the
authoritative pre-temperature-scaling tensor is `out = self.fc(h)`, taken
immediately before `out = self.softmax(out)`; shape `[batch, 2]`, called
`raw_logits` throughout Stage-2 artifacts. The pinned classification trainer
uses `torch.nn.NLLLoss`. No external classification head is authorized.

### 2.3 Data / splits

Official Matbench 5-fold design: every structure is training data in 4 folds
and outer-test data in exactly 1. Official fold 0 and the Stage 0B seed-0
stratified split are immutable. Training/validation/outer-test are pairwise
disjoint (the historical script's apparent duplication of training rows into
validation is *not* reproduced here). See §7 for the authoritative 25-cell
split-hash matrix.

One global, label-free graph preprocessing pass is authorized before model
training (§6) — reads structure dictionaries only, never labels.

### 2.4 Random2-Descriptor intervention

Control and Random2-Descriptor start from one byte-identical native ALIGNN
initialization. Random2 reuses `RANDOM2_CONFIG.json` unchanged: same
intervention, distribution, sample count, **938 steps**, optimizer, learning
rate, RNG offset, deterministic replay rule. The warm-up updates **only
`fc.weight` and `fc.bias`**; the encoder must remain byte-identical (verified
via full non-head state-dict comparison, including BatchNorm running
statistics and `num_batches_tracked`, in `eval()`/`no_grad()`).

Both branches then run exactly **40 supervised epochs, batch size 32**, with
identical epoch-level ordered sample-ID hashes. Python/NumPy/PyTorch
CPU+CUDA RNGs are reset to seed 0 after Random2 warm-up; fresh AdamW +
OneCycleLR are created after that reset. Checkpoint selection: epoch with
minimum validation NLL, earliest epoch wins exact ties. No early
termination.

Grid: **5 official folds × 5 seeds = 25 cells**, 2 branches per cell = 50
trained branches, 4 post-hoc conditions per cell (A_Control_raw,
B_Control_temperature_scaled, C_Random2_Descriptor_raw,
D_Random2_Descriptor_temperature_scaled) = 100 readouts. This exactly matches
`FINAL_AUDIT.json`.

### 2.5 Native raw-logit provenance contract (frozen v17)

"Native raw logits" = exactly the two columns emitted by `ALIGNN.fc` before
`LogSoftmax`. Every forward pass asserts the public ALIGNN result equals
`log_softmax(z)`, and (when labels present) that `NLLLoss(log_softmax(z), y)
== CrossEntropyLoss(z, y)` within `rtol=1e-6, atol=1e-7`. Validation artifacts
must declare `logit_source="ALIGNN.fc output before LogSoftmax"` and
`logit_columns=["z_0","z_1"]`; calibration fails closed on missing/different
provenance. Exported raw logits are never post-LogSoftmax values; the raw
positive-class probability column is audit-only and never fed to temperature
fitting.

### 2.6 Statistical analysis (prospective, frozen 2026-07-15)

One structure = one unit. Concatenate all 5 outer-test folds into one OOF
vector per seed. Compute paired contrasts on aligned structures. **5,000
paired-structure bootstrap repetitions, seed 20260715.** With multiple seeds,
report per-seed OOF effects plus matched-seed/paired-structure hierarchical
intervals; folds are *not* treated as independent replicates. An interval
crossing zero is not described as "supported"; no equivalence claim without a
predeclared margin.

### 2.7 Gates (outcome-neutral)

Before outer-test evaluation: genuine ALIGNN CUDA forward/backward pass on
≥2 periodic structures, native reproducible logit access, exact split
isolation, prospective config hashes, passing unit tests, and all validation
adequacy checks. A failed mandatory gate stops execution — gates are never
weakened to pass. **Validation adequacy checks only basic Control competence
and mechanical integrity; no gate compares Random2 against Control or
requires Random2 to "win."** Any protocol change prompted by fold-0/seed-0
validation evidence requires a new package version and a restart of all
cells (in practice, most later corrections were prompted by *operational*
DelftBlue failures, not scientific ones — see §13).

### 2.8 Overall gate order

package verification → environment setup → static audit → simulation-only
`sbatch --test-only` → A100 preflight → graph cache → smoke → 100-batch
profile → human approval → primary cell (fold 0/seed 0) → paired grid (folds
1–4 × seeds 0–4) → calibration/export → OOF consolidation → final audit.

## 3. Calibration Metric Definitions (frozen v17, unchanged since)

Source: `CALIBRATION_METRIC_DEFINITIONS.md`. Frozen before any outer-test
analysis.

- **NLL:** sample mean two-class cross-entropy from native or
  temperature-scaled `[z_0,z_1]`.
- **ECE-15:** top-label confidence ECE; confidence = `max(p_0,p_1)`,
  correctness = predicted-class indicator; 15 equal-width bins over `[0,1]`,
  bins `[lower,upper)` except the final bin includes 1.
- **Adaptive ECE-15:** top-label confidence ECE over 15 deterministic
  equal-mass groups; stable sort by confidence, split via
  `numpy.array_split`; ties retain sample order.
- **Classwise ECE-15:** equal-width ECE computed separately for class 0
  (`p_0`, `1[y=0]`) and class 1 (`p_1`, `1[y=1]`); reported error is the
  unweighted mean of the two class errors.
- **Brier:** binary positive-class Brier `mean((p_1-y)^2)` — *not* the
  two-class summed Brier score.
- **ROC-AUC:** from continuous positive-class probability `p_1` (v17–v37);
  **as of v38, calculated from the native binary logit margin `z_1-z_0`**
  (rank-equivalent to exact positive-class probability, avoids
  finite-precision sigmoid saturation ties — see §13, v38). Average ranks
  for ties either way; predicted labels are never used as the score.
- **Accuracy/F1:** predicted class = `argmax([z_0,z_1])`; F1 uses class 1 as
  positive.

**Temperature invariance:** predicted class, margin ordering, tie pattern,
and ROC-AUC must be unchanged for every positive shared scalar temperature
(tolerance-aware as of v33/v35/v38 — see §4 and §13).

## 4. Temperature-Scaling Interface Contract (current: v40-authorized, v31 core)

Sources: `TEMPERATURE_SCALING_INTERFACE_CONTRACT.md`,
`V31_NUMERICAL_CONVERGENCE_AMENDMENT.md`,
`V40_POST_TRAINING_NUMERICAL_RESOLUTION_AMENDMENT.md`.

The only approved implementation: **`vendor/muben_temperature_scaling.py`,
SHA-256 `108b3183400725ca53e025c6f0f92690f35069f6fd786091b9d37dc50e90f08b`**,
status `approved_muben_v31_numerical_convergence_amendment` (supersedes the
v26 hash `868654151b87d526f122f741183c23599913541d5d62aada5cb35c91bc510719`,
which remains preserved in provenance, never deleted). MUBen-derived, one
shared scalar; v31 changed only the declared final `log_T` gradient
tolerance, no second algorithm was added.

```text
fit_temperature(validation_logits, validation_labels) -> fitted object
apply_temperature(logits, fitted object) -> scaled logits
```

ALIGNN is one binary task, native logits `[n,2]`, one shared scalar. Fitting
minimizes unweighted inner-validation cross-entropy in float64 over `log_T`
(`T = exp(log_T)`), via LBFGS strong-Wolfe, initial `T=1`, max 500 optimizer
iterations. The fitted record exposes T, objective, convergence, steps,
before/after validation NLL, implementation version, parameterization,
dtype, tolerances, gradient, and source SHA-256.

**Convergence rule, current (v31 baseline):** final absolute `log_T`
gradient ≤ `1e-7`; finite positive T; non-increasing validation NLL within
`1e-12` comparison allowance. (Originally an undocumented `tolerance_grad *
10` multiplier at effective `1e-8` — removed by v31 after fold-0/seed-0 job
`10641918` failed it: Control ended `6.19290274215777e-8`, Random2 ended
`1.622938891964708e-8`.)

**v40 numerical-resolution exception (3 cells only):** for the exact cells
`(fold 1, seed 1)`, `(fold 2, seed 4)`, `(fold 3, seed 1)` — array indices 6,
14, 16 — a fit is *authorized* (not relabeled "converged") when: gradient
≤ `1e-6`; T finite/positive; NLL finite/non-increasing; two independent
refits on the immutable inner-validation artifact return identical fit
metadata/loss curve; source hash matches; fit scope is inner-validation
only. This is recorded as a separate `numerical_resolution_authorized: true`
fact bound to a resolution-record hash — the original `converged` boolean is
never overwritten. Only these 3 exact cells/6 exact validation-artifact
hashes (in `V40_NUMERICAL_RESOLUTION_AUTHORIZATION.json`) are eligible;
anything else fails closed.

**Fatal conditions (always):** fitting on outer-test inputs, fitting
probabilities instead of logits, vector/per-column temperatures, pending
approval, non-positive T, non-finite values, dimension changes, raw-array
mutation, prediction/ranking changes, source-hash mismatch.

**Numerical identity:** `softmax(z/T)[:,1] == sigmoid((z[:,1]-z[:,0])/T)`
within `atol=1e-7, rtol=1e-6`.

**Ordering check (current, v35):** raw margins stably sorted once; scaled
margins in that order must be nondecreasing within `atol=1e-7, rtol=1e-6`
(replaces an earlier exact-permutation/pairwise-tie-matrix check that broke
on floating-point roundoff — see §13, v33/v35).

## 5. Prediction Artifact Schema (current)

Source: `PREDICTION_ARTIFACT_SCHEMA.md`. One row = one structure/condition
record. CSV canonical, UTF-8, comma delimiter, one header, deterministic
official split order.

| Column | Type | Contract |
|---|---|---|
| `structure_id` | string | Official Matbench structure identifier. |
| `fold` | integer | Official outer fold, 0–4. |
| `seed` | integer | Frozen seed, 0–4. |
| `condition` | enum | `A_Control_raw`, `B_Control_temperature_scaled`, `C_Random2_Descriptor_raw`, `D_Random2_Descriptor_temperature_scaled`. |
| `split` | enum | `inner_validation` or `outer_test`. |
| `true_label` | integer | Binary, 0/1. |
| `raw_native_logit_0` | float64 text | Native `fc(h)` logit, class 0, pre-`LogSoftmax`; never reconstructed from probability. |
| `raw_native_logit_1` | float64 text | Native `fc(h)` logit, class 1, pre-`LogSoftmax`; never reconstructed from probability. |
| `raw_probability_positive_audit_only` | float64 text | Stable softmax of the two raw logits; audit only, never fed to temperature fitting. |
| `predicted_label` | integer | `argmax` of native raw logits; positive scalar TS must not change it. |
| `checkpoint_sha256` | lowercase hex | SHA-256 of the selected branch checkpoint. |
| `sample_order_index` | integer | Zero-based position in official split ordering. |

Scaled conditions retain both native raw-logit columns for lineage. Scaled
logits used for metrics live in the calibrated-artifact sidecar (which also
records fitted metadata + approved MUBen source digest) — never substituted
into `raw_native_logit_*`.

Every branch's `validation_raw_logits.npz` must additionally contain
`logit_source="ALIGNN.fc output before LogSoftmax"` and
`logit_columns=["z_0","z_1"]`; calibration fails closed if absent/different.
Forward contract: `model(batch) == log_softmax(z)` and
`NLLLoss(log_softmax(z),y) == CrossEntropyLoss(z,y)` within `rtol=1e-6,
atol=1e-7`.

**Uniqueness key:** `(structure_id, fold, seed, condition, split)`. Within
each `(fold, seed, condition, split)`, `sample_order_index` must be exactly
`0..n-1`; structure IDs and labels must align across all paired conditions.

## 6. Outer-Test Structure Preprocessing Policy (current, from v13)

Source: `OUTER_TEST_STRUCTURE_PREPROCESSING_POLICY.md`. One global,
label-free graph-preprocessing pass is permitted before model training. The
pass reads only each structure dictionary for graph conversion; labels are
never retained, indexed, hashed, copied into the cache, or used to select
graph parameters. The cache manifest stores structure IDs/hashes, graph and
runtime/release provenance, shard locations/offsets/counts, and graph-file
hashes only — never labels.

This authorization is limited to deterministic representation
preprocessing. Outer-test labels/logits/predictions/metrics/decisions remain
prohibited until a cell passes environment, cache, smoke, 100-batch,
training, validation-adequacy, Random2, checkpoint, and final
temperature-approval gates. Cache creation must record
`labels_used_in_graph_construction: false` and exactly one graph
construction per cached structure.

## 7. Split Hash Reconciliation (authoritative 25-cell matrix)

Source: `SPLIT_HASH_RECONCILIATION.md`. `ALL_25_SPLIT_HASHES.json` is the
**runtime authority**: freezes every `(fold, seed)` cell under the executable
policy `random_state=seed`. Contains 25 unique cells, overlap/coverage
assertions, ID hashes, validation-label alignment hashes, and official
outer-test ID hashes — no labels stored. Every training invocation verifies
train IDs, validation IDs, and validation labels against its exact
fold/seed record before graph loading or optimization.

**Historical note (not authoritative):** the older `STAGE_0B_AUDIT.json`
holds one record per fold generated with `random_state=fold` (despite its
prose saying "seed-0"); its 5 records only coincidentally match the
*diagonal* cells `(0,0),(1,1),(2,2),(3,3),(4,4)` of the true 5×5 matrix. An
earlier conditional seed-0 check would have wrongly rejected folds 1–4/seed
0 and left the other 20 cells unchecked. That artifact is preserved as
evidence, not deleted or concealed, but is not used at runtime.

## 8. Resource Profile and Approval Contract (current)

Source: `RESOURCE_PROFILE_APPROVAL_CONTRACT.md`. `FULL_CONFIG_100_BATCH_PROFILE.json`
is non-primary runtime evidence, produced on the certified A100 only after
the exact schema-v2 structure cache passes. `PROFILE_APPROVAL.json` is not
distributed with the package.

Approval requires an identified reviewer, UTC time, decision, reasons; it
binds SHA-256 identities for the profile, A100 certification, package
aggregate, execution plan, cache manifest, resource policy, and primary
Slurm script. **Margins:** conservative paired upper bound must use the
declared safety factor and consume ≤80% of primary wall time; projected
host memory, peak CUDA reserved memory, and projected full-grid disk must
each consume ≤80% of verified capacity.

Any unavailable/missing/non-finite/contradictory/changed input prohibits
approval. A rejected/copied/stale/mismatched record blocks primary, grid,
calibration, OOF, and final-audit jobs. Resource or identity changes require
a new profile + approval; packaged-file changes require a new additive
release. Changing wall time, CPU count, host memory, GPU request, cache
identity, package identity, execution plan, or resource policy invalidates
an existing approval.

## 9. Environment & Dependency Setup (current state)

Sources: `ENVIRONMENT_SETUP_CORRECTION.md` (v16),
`DEPENDENCY_RESOLUTION_AUDIT.md`, `DELFTBLUE_PACKAGE_READINESS_REPORT.md`,
`AUDIT_REPORT.md`, `DELFTBLUE_RUNBOOK.md`.

- **Target environment prefix:**
  `/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9`
  (revision-9 environment; `.staging` sibling used during build).
- **Target platform:** CPython 3.10.20, Linux x86-64, glibc 2.28/2.35,
  pip 25.3 (pinned before any production dependency stage).
- Every enumerated remote distribution is version- and SHA-256-pinned,
  installed `--no-deps --require-hashes`; the full graph is checked with
  `pip check`. Hash locks: `BOOTSTRAP_REQUIREMENTS.txt`,
  `TORCH_CU118_REQUIREMENT.txt`, `DGL_CU118_REQUIREMENT.txt`,
  `CUDA11_RUNTIME_REQUIREMENTS.txt`, `PYTHON_DEPENDENCY_LOCK.txt`.
  Python-3.10-only pytest deps `exceptiongroup==1.3.1` and `tomli==2.4.1`
  are explicitly hash-pinned (their absence caused the real v15 DelftBlue
  failure that prompted v16). `importlib-metadata` excluded (marker <3.10);
  Windows-only `colorama` removed as unreachable;
  `typing-extensions==4.16.0` selected. Combined closure: 79 distributions /
  83 wheel files (CPython-3.10, Linux).
- **CUDA graph:** runtime 11.8.89, cuBLAS 11.11.3.6, cuSPARSE 11.7.5.86,
  cuSOLVER 11.4.1.48. The invalid historical requirement
  `nvidia-nvjitlink-cu11==11.8.86` is absent by design (CUDA-11 cuSPARSE has
  no nvJitLink dependency). Login certification loads `libcudart.so.11.0`,
  `libcublas.so.11`, `libcusparse.so.11`, `libcusolver.so.11`, then imports
  torch/DGL/ALIGNN without a GPU.
- **DelftBlue login shell caveat:** default login Python is 3.6.8 and cannot
  parse `from __future__ import annotations`. Bootstrap verification
  therefore explicitly uses Python 3.10.12 inside an isolated subshell
  (`module load python/3.10.12`); `module purge` must never be run directly
  in the long-lived parent login shell (an isolated subshell's `module
  purge` doesn't strip the parent's Slurm commands, so run it only inside
  the subshell).
- **Certification ordering (non-circular, fixed v18):** static resolution
  proof checked before installation → staging install/test in explicit
  bootstrap mode → Conda prefix-aware clone to final prefix → **force-replay
  of the hash-locked bootstrap tools after cloning** (fixed v20, because
  Conda's clone silently overlaid pip 26.2.1/setuptools 84.0.0/wheel 0.47.0
  over the pinned pip 25.3/setuptools 80.9.0/wheel 0.45.1) → final prefix
  re-verified → real clean-install evidence + transcript written beside the
  final prefix (outside the immutable package) → full pytest run against
  that external evidence → atomic login certificate binding package
  aggregate + evidence + transcript + pytest-log hashes.
  `CLEAN_INSTALL_EVIDENCE.json` / `CLEAN_INSTALL_TRANSCRIPT.txt` inside the
  immutable package are deliberately pending placeholders and can never
  themselves certify a runtime; a partial final prefix or stale staging
  prefix is rejected.
- Every GPU Slurm script sources one shared CUDA-activation helper
  (`scripts/activate_cuda_runtime.sh`); the two "compute-only" scripts
  remain intentionally uncoupled from it.

## 10. Execution Gate Order & Current Runbook

Source: `DELFTBLUE_RUNBOOK.md` (still the operative stage-by-stage runbook;
v41 only changes the OOF/final-audit evidence-path stage, see §1.1 and
`V41_OOF_RUNBOOK.md`).

Stage 0 — A100 runtime certification (`slurm/00_a100_preflight.sbatch`,
require `ALIGNN_ENVIRONMENT: PASS`) → Stage 1 — label-free graph cache
construction + exhaustive verification (`slurm/05_build_graph_cache.sbatch`,
require `ALIGNN_GRAPH_CACHE: PASS`; invalid/partial/stale shards moved to an
explicit cache-quarantine root, never silently deleted) → Stage 2a — smoke
(`slurm/06_smoke_fold0_seed0.sbatch`; first runs
`scripts/verify_execution_contracts.py --device cuda` on synthetic tensors:
singleton final batch, pre-LogSoftmax loss identity, resumed-vs-uninterrupted
equivalence, exact OneCycle step count, paired initial hashes, BatchNorm-state
immutability, CPU/CUDA RNG restoration, sample-order generator restoration,
ranking/class/AUROC invariance — none of this authorizes primary training) →
Stage 2b — 100-batch resource profile + manual review/approval
(`slurm/07_profile_100_batches.sbatch`, `scripts/review_profile.py`, see §8)
→ Stage 2c — primary cell fold0/seed0, manual submission only, explicitly
**no** `afterok` dependency on the profile
(`bash scripts/require_profile_current.sh …` then
`slurm/08_primary_fold0_seed0.sbatch`; require
`ALIGNN_FOLD0_SEED0_PRIMARY: PASS`) → Stage 3 — paired grid + calibration
(`slurm/10_train_fold_seed.sbatch`, `slurm/20_calibrate_export.sbatch` via
`afterok`; failed indices retried individually; verify with
`scripts/audit_grid_complete.py`) → Stage 4 — OOF consolidation + final audit
(`slurm/30_consolidate_oof.sbatch` → `afterok` →
`slurm/40_final_audit.sbatch`; final line must read
`ALIGNN_PERSISTENT_FINAL_AUDIT: PASS`).

**v41 current entry point** (post-v40, all 25 cells complete): bind the
correct evidence roots and submit only Stage 4:

```bash
export ALIGNN_ENV_PREFIX="/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9"
export ALIGNN_V26_PACKAGE_ROOT="$HOME/alignn_stage2_v26/delftblue_package_v26"
export ALIGNN_V36_PACKAGE_ROOT="$HOME/alignn_stage2_v36/delftblue_package_v36"
export ALIGNN_GRAPH_CACHE_ROOT="/scratch/$USER/alignn_matbench_is_metal_graph_cache_v26_verified"
export ALIGNN_WORK_ROOT="$HOME/ml4md/ALIGNN/stage2_primary_v26"
export ALIGNN_OOF_OUTPUT="$ALIGNN_WORK_ROOT/oof_analysis_v41"

OOF_JOB=$(sbatch --parsable --export=ALL slurm/30_consolidate_oof.sbatch); OOF_JOB="${OOF_JOB%%;*}"
AUDIT_JOB=$(sbatch --parsable --dependency=afterok:$OOF_JOB --export=ALL slurm/40_final_audit.sbatch)
```
Require `ALIGNN_OOF_CONSOLIDATION: PASS` then `ALIGNN_PERSISTENT_FINAL_AUDIT: PASS`
(both already satisfied per §1 — this is documented for reproducibility, not
as a pending action).

Upload/verify pattern used throughout v37–v41 (PowerShell → DelftBlue):
```powershell
$Base = "C:\Users\User\OneDrive - Delft University of Technology\Master Y1\Q4\Research\experiments\matbench_alignn_is_metal_calibration\stage2"
scp "$Base\alignn_stage2_delftblue_vNN.tar.gz" "hhew@login.delftblue.tudelft.nl:~/"
scp "$Base\DELFTBLUE_ARCHIVE_MANIFEST_VNN.json" "hhew@login.delftblue.tudelft.nl:~/"
```
```bash
echo "$(python3 -c 'import json; print(json.load(open("DELFTBLUE_ARCHIVE_MANIFEST_VNN.json"))["archive_sha256"])')  alignn_stage2_delftblue_vNN.tar.gz" | sha256sum --check
```

## 11. MUBen / Uni-Mol Temperature-Scaling Evidence (final state)

Sources: `MUBEN_TS_EVIDENCE_AUDIT.md`, `MUBEN_TS_INTEGRATION_CHECKLIST.md`,
`UNIMOL_TS_RECONCILIATION_CHECKLIST.md`.

Status: `approved_muben_persistent_final_audit_passed`. The Uni-Mol evidence
audit reports **PASS for all 6 expected cells** at release commit
`446471d46449ec79cc0f846baa54d7f4a2f90695`; source SHA-256
`f4989a94700990e768cbe595a1399f23dc79e33df2471dd4e12ec32bf5245d43`; completed
evidence archive SHA-256
`410e96c2257562601646705e4fe6912413bf69d51b9dd9296bbd2c405e4c9c19`.

MUBen's exact reproduction uses its native (unconstrained) optimizer; ALIGNN
does not claim that as its own protocol — it requires the one-task shared
scalar, float64 unweighted validation NLL, numerical-convergence adapter
described in §4, pinned at SHA-256
`108b3183400725ca53e025c6f0f92690f35069f6fd786091b9d37dc50e90f08b`. All
integration-checklist items are checked off: audit complete, hash frozen,
extension implemented, pending/generic/mismatched approvals rejected,
softmax/sigmoid equivalence + determinism + finite-T + raw-immutability +
ranking-invariance + extreme-logit checks verified, Control and Random2
fit separately from their own cached inner-validation logits. No further
reconciliation is required unless a new scaler source is independently
audited and adopted as another additive package version.

## 12. Artifacts (v41 current outputs)

See §1 for the authoritative table. In short: `OOF_ANALYSIS.json` (completed
OOF + clustered bootstrap) and `FINAL_AUDIT.json` (25/25 cells, 50 branches,
100 readouts, 0 failures) are the two headline result files, both included
in the package; the two large row-level artifacts (`oof_predictions.csv`,
`alignn_descriptor_v41_final_results.tar.gz`) live outside git at the paths
and hashes given in §1.

## 13. Version History / Changelog (condensed)

Only major pivots get more than one line. Every correction below was
triggered by a **real observed DelftBlue failure or test-suite gap**, never
by outer-test outcomes — the scientific gates remained outcome-neutral
throughout (§2.7). Gaps in the numbering (v1–v4, v6, v8–v9, v21–v24, v27–v30)
correspond to versions with no standalone report in this doc set.

| Version | Summary |
|---|---|
| v5 | Root cause of a DelftBlue manifest-hash mismatch: OS-dependent path sort order (Windows vs Linux) changed the hash-concatenation order. Also found stray `.pyc`/`__pycache__` debris. Fixed by v6 (not separately documented) via canonical POSIX-sorted hashing. |
| v7 | Test report only: 29/29 passed in both working package and clean extraction; added installed-source provenance and vendored-determinism checks. |
| v10 | 47 passed. Coverage expanded to approval hashing, LBFGS convergence, split hashes, staged gate scripts, cache reuse, pip isolation. |
| v11 | 62 passed. Added cache-provenance, corruption-quarantine/rebuild, interrupted-recovery, and profile-approval CLI coverage. |
| v12 | 75 passed. Fixed DelftBlue Python-3.6.8-in-login-shell incompatibility via isolated Python-3.10.12 module subshell; added `gpu-a100` Slurm headers and memory-unit parsing. |
| v13 | 89 passed. Added Slurm partition memory-limit checks for all 9 scripts, non-exclusive-node rejection, mocked `--test-only` harness. |
| v14 | 100 passed. Added CUDA-runtime regression tests (missing cuSPARSE / search path), exact CUDA component pinning, shared GPU-activation-helper enforcement across all 7 GPU scripts. |
| v15 | 108 passed. Generated real online dependency-resolution evidence (79 distributions / 83 wheels for Linux CPython 3.10); confirmed hash-locked, no-deps replay. |
| v16 | **Environment-setup correction.** Real v15 DelftBlue failure: pytest's Python-3.10-only `exceptiongroup>=1` marker dependency was absent from the hash lock. Fixed by pinning pip to 25.3 first and hash-locking `exceptiongroup`/`tomli`. See §9. |
| **v17** | **Major pivot — Scientific Integrity Amendment.** Froze native raw-logit provenance (§2.5), BatchNorm/encoder immutability for Random2 descriptor extraction, calibration metric definitions (§3), the Delta_s primary estimand and structure-clustered bootstrap (§2.6), and made validation gates outcome-neutral (§2.7). This is the scientific baseline every later version preserves. |
| v18 | **Bootstrap-certification amendment.** Fixed a circular gate: setup previously required a passed clean-install artifact before it could create one. Separated static resolution proof (checked pre-install) from runtime evidence (generated post-install, outside the immutable package) and bound both into an atomic login certificate. |
| v19 | Fixed a real v18 DelftBlue failure: `verify_environment.py` couldn't import the package-local `alignn_stage2` module (no `PYTHONPATH`). Fixed via explicit package-root propagation + self-bootstrapping import discovery. Scientific run unaffected (failed before any training). |
| v20 | Fixed a real v19 failure: Conda's prefix-aware clone silently overlaid newer pip/setuptools/wheel over the pinned hash-locked bootstrap versions. Fixed by force-replaying the bootstrap lock after cloning, before final verification. |
| v25 | Fixed a real DelftBlue profile-job crash (job `10633397`): a local variable named `output` was reused for both the report path and the model tensor, causing `AttributeError: 'Tensor' object has no attribute 'parent'` right before the atomic write. Renamed to `output_path`/`model_output`. |
| v26 | Fixed a false-negative regression test left over from v25 (a forbidden-substring check incidentally matched the *correct* fixed identifier `model_output`). No behavior change. |
| **v31** | **Major pivot — Numerical Convergence Amendment.** Real fold-0/seed-0 job `10641918` failed LBFGS convergence at an undocumented effective `1e-8` gradient threshold (actual: Control `6.19e-8`, Random2 `1.62e-8`). Declared tolerance relaxed to `1e-7` (post-training, pre-outer-test, based only on numerical diagnostics, never on outcomes). New MUBen adapter hash `108b3183…` supersedes v26 hash `868654151…`. See §4. |
| v32 | Fixed a real Slurm rejection: `gpu-a100-small` allows ≤2 CPUs/task, but the recovery job requested 8. Moved that job to regular `gpu-a100`. |
| v33 | Fixed a real recovery-job failure (`10643036`): an exact-`argsort`-permutation equality check on temperature-scaled margins broke on floating-point tie-order noise. Replaced with a tolerance-aware monotonicity check (`atol=1e-7, rtol=1e-6`). See §4. |
| v34 | Fixed a stale test asserting 7 `gpu-a100` scripts should exist (correct count: 8). No behavior change. |
| v35 | **Production-invariance recovery.** Real job `10644192` completed its single authorized outer-test export, then failed closed in `verify_cell.py` because the *production* verifier hadn't yet adopted the v33 tolerance-aware ordering check. Fixed, and added logic so that when an outer-test sentinel already exists, recovery reuses the existing export rather than re-materializing it (single-use-export invariant). |
| v36 | Fixed 2 stale Slurm CPU-count assertions that didn't account for v35's 2-CPU `gpu-a100-small` recovery job. **The fold-0/seed-0 primary cell reached full PASS here** — this is the "v36 primary gate" referenced by every later version. |
| v37 | **Full-grid submission.** Submitted the remaining 24 training cells (array indices 1–24; index 0 already complete), with calibration chained via `afterok`. No scientific or calibration-implementation change. |
| **v38** | **Major pivot — result-blind numerical recovery, after all 24 cells trained.** (a) Fixed AUROC to use the native binary margin instead of exact sigmoid probability, because finite-precision sigmoid saturation created spurious exact-equality ties (§3). (b) Recovered 5 cells stuck at the existing-export boundary — array indices 9,10,15,17,19 = `(1,4),(2,0),(3,0),(3,2),(3,4)` — without recomputing anything. (c) Recorded but left **blocked** 3 cells that failed inner-validation convergence diagnostics — array indices 6,14,16 = `(1,1),(2,4),(3,1)` — status `blocked_pending_validated_muben_numerical_resolution`. All of this used only process states/failure types/artifact presence — no outer-test row or metric was inspected. |
| v39 | Fixed one stale test assertion still requiring pre-v38 AUROC wording (`continuous positive-class probability` → `native binary logit margin`). No production-code change. |
| **v40** | **Major pivot — numerical resolution of the final 3 blocked cells** (`(1,1),(2,4),(3,1)`, authorized 2026-08-16). New rule: gradient ≤`1e-6` (loosened from `1e-7`), T finite/positive, NLL non-increasing, two independent refits give identical metadata, exact source-hash match — applied *only* to these 3 exact cells/6 exact validation-artifact hashes; original `converged` flags never overwritten (§4). After this, all **25/25 cells complete**. |
| **v41** | **Current.** OOF-evidence-binding correction only — repoints OOF consolidation and final audit at the correct v26/v36/v37 evidence chain plus a fresh 25-cell hash audit; no computation changed (§1.1). Result: `FINAL_AUDIT.json` PASS with 25 valid cells / 50 branches / 100 readouts / 0 failures, `OOF_ANALYSIS.json` complete. |

## 14. Open Issues, Caveats, Known Limitations

- **No stated scientific conclusion in these docs.** All 62 files describe
  process integrity (splits, provenance, calibration mechanics, resource
  gates, recovery) — none reports the actual sign/magnitude of `Delta_s` or
  whether Random2-Descriptor helped. That answer requires opening
  `OOF_ANALYSIS.json` / the final results archive directly (paths in §1).
- **Post-hoc numerical-tolerance relaxations are real and documented, not
  hidden.** The LBFGS convergence gradient tolerance moved `1e-8`
  (undocumented) → `1e-7` (v31) → `1e-6` for exactly 3 cells (v40). Each
  relaxation was made *after* training but *before* outer-test access, using
  only numerical diagnostics — the docs are explicit that this is not the
  originally prespecified protocol and say so plainly. A careful reader
  double-checking robustness should treat cells `(1,1)`, `(2,4)`, `(3,1)` as
  numerically weaker than the other 22 (only `1e-6` gradient tolerance, vs
  `1e-7` elsewhere), and should be aware two of the AUROC/ordering checks
  were themselves buggy for a period (v33/v35/v38) before being fixed.
- **5 cells recovered via existing-export reuse, not fresh computation**:
  indices 9,10,15,17,19 = `(1,4),(2,0),(3,0),(3,2),(3,4)` (v38 Path B). Their
  outer-test export was produced once, under the pre-v38 AUROC definition,
  and never recomputed — only re-verified under the corrected metric
  definitions and re-promoted.
- **Historical `STAGE_0B_AUDIT.json` remains on disk but is not the runtime
  split authority** (§7) — a future reader must use `ALL_25_SPLIT_HASHES.json`,
  not the older file, or risk checking the wrong 5 of 25 cells.
- **Local Windows machine cannot fully verify the pipeline.** All "local"
  test reports (v7–v40) explicitly state they never install/run the real
  DGL/ALIGNN/CUDA/Slurm stack; full runtime pytest and `sbatch --test-only`
  remain DelftBlue-only operator gates. Treat local package-build "N passed"
  counts as necessary but not sufficient.
- **Large artifacts are hash-pinned but not committed to git** (§1) — anyone
  needing row-level reproducibility must fetch them from the recorded
  workspace paths and verify against the recorded SHA-256 before using them.
- **`STAGE_2_PROTOCOL_BLOCKER.md`** is referenced by
  `STAGE_2_PROSPECTIVE_AMENDMENT.md` as a blocker that amendment resolves,
  but that blocker file itself is not present among the 62 files reviewed
  here (likely superseded/removed) — treat the amendment's text as the
  current authority regardless.

## 15. Source File Index

All facts above are drawn from the 62 files below (`descriptor_v41/` and
`descriptor_v41/package/`); none were modified in producing this summary.

Protocol/contract/policy/schema: `EXPERIMENT_PROTOCOL.md`,
`RESOURCE_PROFILE_APPROVAL_CONTRACT.md`,
`TEMPERATURE_SCALING_INTERFACE_CONTRACT.md`,
`CALIBRATION_METRIC_DEFINITIONS.md`, `PREDICTION_ARTIFACT_SCHEMA.md`,
`OUTER_TEST_STRUCTURE_PREPROCESSING_POLICY.md`,
`ENVIRONMENT_SETUP_CORRECTION.md`, `PROSPECTIVE_ANALYSIS_PLAN.md`,
`STAGE_2_PROSPECTIVE_AMENDMENT.md`, `SPLIT_HASH_RECONCILIATION.md`.

Runbooks/reports: `README.md`, `DELFTBLUE_RUNBOOK.md`,
`DELFTBLUE_PACKAGE_READINESS_REPORT.md`, `V41_OOF_RUNBOOK.md`,
`V41_OOF_EVIDENCE_BINDING_CORRECTION.md`, `ARTIFACTS.md` (parent-level).

Audit/analysis: `AUDIT_REPORT.md`, `DEPENDENCY_RESOLUTION_AUDIT.md`,
`MUBEN_TS_EVIDENCE_AUDIT.md`, `MUBEN_TS_INTEGRATION_CHECKLIST.md`,
`UNIMOL_TS_RECONCILIATION_CHECKLIST.md`.

Version chain: `V5_MANIFEST_ROOT_CAUSE.md`, `V7_TEST_REPORT.md`,
`V10_TEST_REPORT.md` … `V20_TEST_REPORT.md`,
`V19_IMPORT_PATH_CORRECTION.md`, `V20_CONDA_CLONE_OVERLAY_CORRECTION.md`,
`V25_PROFILE_OUTPUT_CORRECTION.md`, `V26_TEST_CORRECTION.md`,
`V31_NUMERICAL_CONVERGENCE_AMENDMENT.md`, `V31_AUDIT_REPORT.md`,
`V31_RECOVERY_RUNBOOK.md`, `V32_RESOURCE_CORRECTION.md`,
`V32_RECOVERY_RUNBOOK.md`, `V33_ORDERING_TOLERANCE_CORRECTION.md`,
`V33_RECOVERY_RUNBOOK.md`, `V34_TEST_COUNT_CORRECTION.md`,
`V34_RECOVERY_RUNBOOK.md`, `V35_PRODUCTION_INVARIANCE_RECOVERY.md`,
`V35_TEST_REPORT.md`, `V35_RECOVERY_RUNBOOK.md`,
`V36_STALE_RESOURCE_TEST_CORRECTION.md`, `V36_TEST_REPORT.md`,
`V37_FULL_GRID_SUBMISSION.md`, `V38_NUMERICAL_RECOVERY_AMENDMENT.md`,
`V38_RECOVERY_RUNBOOK.md`, `V39_STALE_AUROC_TEST_CORRECTION.md`,
`V39_RECOVERY_RUNBOOK.md`,
`V40_POST_TRAINING_NUMERICAL_RESOLUTION_AMENDMENT.md`,
`V40_TEST_REPORT.md`, `V40_RECOVERY_RUNBOOK.md`,
`V17_IMPLEMENTATION_REPORT.md`, `V17_SCIENTIFIC_INTEGRITY_AMENDMENT.md`,
`V17_TEST_REPORT.md`, `V18_BOOTSTRAP_CERTIFICATION_AMENDMENT.md`,
`V18_TEST_REPORT.md`.
