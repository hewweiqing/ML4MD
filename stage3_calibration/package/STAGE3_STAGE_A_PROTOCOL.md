# Stage A — initialization diagnostic (frozen protocol)

## Why

`descriptor_v41` and `coordinate_gpu_v4`'s Random2 warm-up variants only
make sense as evidence for Cheon & Paik's (Nat Mach Intell 2026) mechanism
if a randomly-initialized ALIGNN is actually overconfident at init. For a
2-class SoftMax, chance is exactly 0.5 max-softmax. This has never been
measured in this repo. Stage A measures it, and gates everything else: if
there is no initial overconfidence, the warm-up cannot be operating via the
paper's mechanism, and that is reported as a structural negative rather
than smoothed over.

## Must be run on DelftBlue

`alignn`/`dgl`/CUDA are not available on the machine this code was written
on. `init_diagnostic.py` is written and structurally reviewed but its
measurements have never been executed. The first real run must be preceded
by dumping the full `ALIGNN(model_config())` `named_modules()` name/type
inventory to `ALIGNN_MODULE_INVENTORY.json` (done automatically by
`scripts/run_stage_a_diagnostic.py`) — the activation-RMS trace is recorded
by execution-call-order index regardless, but semantic module labels need
that one-time inventory.

## Submission sequence (`slurm/`)

Three jobs, in order, gated by `scripts/require_stage_a_gates.py`:

1. `00_stage_a_a100_preflight.sbatch` — CUDA/A100 runtime + a minimal
   head-only optimizer step. Reuses `coordinate_gpu_v4`'s certified
   conda environment (`activate_cuda_runtime.sh`, copied verbatim — same
   version pins, same environment prefix).
2. `05_profile_full_network_seed0.sbatch` — times **one** (fold 0, seed 0)
   full-network warm-up run before committing to the full 20-seed job. 938
   whole-network optimizer steps run twice (deterministic-replay contract)
   is a much larger intervention than the existing head-only arms, so its
   cost is measured, not assumed. Writes
   `preflight/STAGE_A_FULL_NETWORK_SEED0_PROFILE.json`. A human reviewer
   must then write a matching `STAGE_A_FULL_NETWORK_SEED0_PROFILE_APPROVAL.json`
   (`schema_version`, `decision: "approved"`, `reviewer_identity`, `reasons`,
   `approved_at_utc`, `bindings` from `stage_a_profile.binding_values(...)`,
   `resource_policy_evaluation` from `stage_a_profile.evaluate_policy(...)`
   against `STAGE_A_RESOURCE_POLICY.json`) before the next step is unlocked
   — `require_stage_a_gates.py` checks this via `stage_a_profile.verify_approval()`.
3. `10_stage_a_full_fold0.sbatch` — the full 20-seed Stage A run (fold 0,
   single job, not an array — one process loops all seeds). Requires both
   the preflight and the profile approval to pass first.

`resource_profile.py`'s schema from `coordinate_gpu_v4` was deliberately
**not** reused here — it encodes assumptions specific to that package's own
25-cell paired-branch training grid (persisted shard cache accounting,
control/random2 branch-hour projections) that don't describe what Stage A
actually measures. `stage_a_profile.py` follows the same
validate/evaluate-against-policy/require-human-approval pattern with a
schema that matches the single-seed timing report this package actually
produces.

## Model

`ALIGNNConfig(name="alignn", alignn_layers=4, gcn_layers=4,
atom_input_features=92, edge_input_features=80, triplet_input_features=40,
embedding_features=64, hidden_features=256, classification=True,
num_classes=2, link="identity")` — identical to every other package in this
repo. 20 seeds (0-19), `seed_all(seed)` before each construction.

## Three input sets, >=2000 structures each

- **(a) Real**: first >=2000 inner-training structures (fold-scoped, from
  the frozen `ALL_25_SPLIT_HASHES.json`-verified split) by deterministic
  SHA-256 ordering.
- **(b) Coordinate-perturbed**: same structures, `sigma=0.04` Angstrom per
  Cartesian axis via `coordinate_noise.perturb_structure` — the exact same
  mechanism as `coordinate_gpu_v4`'s real arm, same authorized sigma value
  (reused, not re-selected; see `GPU_COORDINATE_V29_AUTHORIZATION_BINDING_SOURCE.md`).
- **(c) Random-feature**: same graph topology, node/edge/angle feature
  tensors resampled from N(0,1).

## Measurements, per (seed, input set)

- Logit mean/std/min/max, `|z1-z0|` gap, max-softmax mean/histogram/
  fraction>0.9, predictive entropy, predicted-class-1 fraction — computed
  over the full input set (all ~2000 structures), in `measure()`.
- **Raw per-sample native logits** for every (seed, phase, input_set)
  triple, float32, shape `[n,2]`, written to a compact npz sidecar
  (`STAGE_A_RAW_LOGITS_fold{fold}.npz`, key format
  `seed{seed}__{phase}__{input_set}`) rather than embedded in the indented
  JSON report — added after an audit found the report only ever carried
  aggregate statistics and one coarse 10-bin confidence histogram, which
  cannot support a real logit histogram or a logit-vs-softmax scatter. Size
  estimate: 20 seeds x 3 input sets x 5 phases x 2000 samples x 2 floats x
  4 bytes (float32) ~= 4.8MB per fold — fine as a binary sidecar, would have
  been wasteful as JSON text.
- Per-leaf-module activation RMS from **one representative batch**
  (`ACTIVATION_PROBE_BATCH_SIZE = 32`), not the full input set — see
  `measure_activation_scale()`'s docstring in `activation_probe`-adjacent
  code (`init_diagnostic.py`) for why: an earlier version reused a single
  `ActivationTrace` across every batch of the full-dataset loop (~63
  batches for 2000 structures) without resetting between them, so
  `call_order_index` kept incrementing across batches and
  `rms_first`/`rms_last`/`monotonic_non_increasing` were silently computed
  over ~63 concatenated depth traversals rather than one clean pass —
  populated, plausible, and wrong, in the figure carrying the paper's
  central structural claim about where normalization pins activation
  scale. A single representative batch removes the reset-correctness
  question entirely rather than solving it; characterizing RMS-across-depth
  never needed 2000 structures. BatchNorm modules are flagged explicitly at
  write time (`is_batchnorm`, not recomputed later) — ALIGNN uses
  BatchNorm throughout, so a strictly monotonic "gradual reduction of
  activation scale across the hierarchy" is not structurally expected here
  regardless of what any single trace shows.
- Every measurement record carries explicit `"phase"` and `"input_set"`
  fields (not just JSON-nesting-as-identity), including for
  `head_perturbation_control`, so a flattener doesn't have to special-case
  one output shape against the others.

## Decision criterion

`mechanism_verdict` in `activation_probe.py`, one of:
- `overconfidence_present` — mean max-softmax >= 0.60 and fraction above
  0.9 >= 0.05.
- `no_initial_overconfidence` — mean max-softmax <= 0.55 and mean logit gap
  <= 1.0.
- `ambiguous_see_full_distribution` — neither threshold cleanly met; an
  honest third outcome rather than a forced binary call.

These thresholds are fixed and documented at the top of `activation_probe.py`,
not tuned against any measured outcome (none exists in this environment).

## Repeated after four conditions

1. **Descriptor** head-only warm-up (`descriptor_head_warmup.py`).
2. **Coordinate** head-only warm-up (`coordinate_head_warmup.py`).
3. **Full-network** (SCRATCH) warm-up (`full_network_warmup.py`,
   `FULL_NETWORK_WARMUP_CONFIG.json`).
4. **Head-perturbation control** — no noise training at all: `fc.weight`/
   `fc.bias` displaced by an independent random direction matched in L2
   norm to what the descriptor warm-up produced for the same seed. This is
   the control that distinguishes "the warm-up mechanism specifically did
   something" from "any head displacement of this size does the same
   thing." Reported separately, never merged with the three real arms.

## Mechanical gate vs. scientific finding

The existing `[0.45,0.55]` mean-positive-probability / `>=0.68` entropy
post-warm-up gate (`evaluate_mechanical_gate`) is a pipeline precondition
only — it proves the warm-up did *something* non-degenerate. It is never
read by `mechanism_verdict` and is never cited as evidence for the paper's
mechanism.
