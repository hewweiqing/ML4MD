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
  fraction>0.9, predictive entropy, predicted-class-1 fraction.
- Per-leaf-module activation RMS in true forward-execution order (hook
  fires in call order regardless of any assumption about ALIGNN's internal
  names), flagged explicitly for BatchNorm modules — ALIGNN uses BatchNorm
  throughout, so a strictly monotonic "gradual reduction of activation
  scale across the hierarchy" is not structurally expected here regardless
  of what any single trace shows.

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
