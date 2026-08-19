# stage3_calibration runbook

Practical "how do I actually run this" guide. For the technical
spec (what's measured, why, decision thresholds), see
[`package/STAGE3_STAGE_A_PROTOCOL.md`](package/STAGE3_STAGE_A_PROTOCOL.md).
This file lives outside `package/` on purpose — it's a living doc, not part
of the hash-verified immutable tree, so it can be updated without
regenerating `PACKAGE_MANIFEST.json`.

## What this runs

Stage A: a 20-seed initialization diagnostic (is a randomly-initialized
ALIGNN actually overconfident?) plus the new Random2-Full whole-network
warm-up and a head-perturbation control. No supervised training happens —
Stage B/C don't exist yet.

## Before you start

- Requires DelftBlue (or an equivalent node with the pinned `alignn`/`dgl`/
  CUDA A100 stack) — see `package/DELFBLUE_CU118_CONSTRAINTS.txt` and
  friends for the exact pins.
- `coordinate_gpu_v4` and `descriptor_v41` are **frozen** — do not edit
  either package while `coordinate_gpu_v4`'s 24 pending cells are in
  flight (see `coordinate_gpu_v4/STATUS.md`). Nothing here requires
  touching them.
- Sanity check first, if you haven't already: `python verify_packages.py`
  from the repo root should print `All 3 immutable package manifests
  passed.`

## Step 1 — A100 preflight

```bash
cd stage3_calibration/package
sbatch slurm/00_stage_a_a100_preflight.sbatch
```

Writes `preflight/STAGE_A_A100_PREFLIGHT.json`. Must show
`"status": "passed"` before continuing.

## Step 2 — time one seed of the full-network arm

938 whole-network optimizer steps run *twice* (deterministic-replay
contract) is a much bigger intervention than the head-only arms, so its
cost gets measured before committing to the full 20-seed job, not assumed.

```bash
export ALIGNN_DATASET=/path/to/matbench_mp_is_metal.json.gz
sbatch slurm/05_profile_full_network_seed0.sbatch
```

Writes `preflight/STAGE_A_FULL_NETWORK_SEED0_PROFILE.json`, including
`elapsed_seconds`, `projected_seconds_for_20_seeds`, and
`deterministic_replay`.

## Step 3 — write the approval JSON (human step, not automated)

Read the profile from step 2. If `projected_seconds_for_20_seeds / 3600`
is comfortably under `STAGE_A_RESOURCE_POLICY.json`'s
`projected_wall_time_hours_max` (12h by default), write
`preflight/STAGE_A_FULL_NETWORK_SEED0_PROFILE_APPROVAL.json`:

```python
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone

profile_path = Path("preflight/STAGE_A_FULL_NETWORK_SEED0_PROFILE.json")
manifest = json.loads(Path("PACKAGE_MANIFEST.json").read_text())
profile = json.loads(profile_path.read_text())
policy = json.loads(Path("STAGE_A_RESOURCE_POLICY.json").read_text())

bindings = {
    "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
    "package_aggregate_sha256": manifest["aggregate_sha256"],
}
projected_hours = profile["projected_seconds_for_20_seeds"] / 3600
evaluation = {"passed": projected_hours <= policy["projected_wall_time_hours_max"],
    "projected_wall_time_hours": projected_hours,
    "projected_wall_time_hours_max": policy["projected_wall_time_hours_max"]}

approval = {"schema_version": 1, "decision": "approved",
    "reviewer_identity": "<your name>", "reasons": ["projected time within budget"],
    "approved_at_utc": datetime.now(timezone.utc).isoformat(),
    "bindings": bindings, "resource_policy_evaluation": evaluation}
Path("preflight/STAGE_A_FULL_NETWORK_SEED0_PROFILE_APPROVAL.json").write_text(
    json.dumps(approval, indent=2, sort_keys=True))
```

`require_stage_a_gates.py` (called automatically by the next step) checks
this against `alignn_stage2/stage_a_profile.py`'s `verify_approval()` —
wrong bindings or a failed policy evaluation raises and blocks the run.

## Step 4 — the full 20-seed run

```bash
export ALIGNN_DATASET=/path/to/matbench_mp_is_metal.json.gz
export ALIGNN_STAGE_A_WORK_ROOT=/path/to/output/dir
export ALIGNN_COORDINATE_SIGMA_AUTHORIZATION=/path/to/verified_coordinate_sigma_authorization.json
sbatch slurm/10_stage_a_full_fold0.sbatch
```

`ALIGNN_COORDINATE_SIGMA_AUTHORIZATION` must point at the *same*
already-verified artifact `coordinate_gpu_v4` uses (same sigma=0.04Å
value, not re-selected) — see
[`package/GPU_COORDINATE_V29_AUTHORIZATION_BINDING_SOURCE.md`](package/GPU_COORDINATE_V29_AUTHORIZATION_BINDING_SOURCE.md).

On its first invocation, `run_stage_a_diagnostic.py` also dumps
`ALIGNN_MODULE_INVENTORY.json` (full `named_modules()` name/type list) —
worth diffing against the structure already confirmed and documented in
`STAGE3_STAGE_A_PROTOCOL.md`'s "real-environment verified" section (105
leaf modules, `readout_feat` expected to never fire) to catch any
surprise if the cluster's `alignn` build differs from the local one this
was validated against.

Output: `$ALIGNN_STAGE_A_WORK_ROOT/STAGE_A_DIAGNOSTIC_fold0.json` and
`STAGE_A_RAW_LOGITS_fold0.npz`.

## What to check when results land

Pulled straight from the report, no manual computation needed:

- `cross_seed_verdict_summary` — is the verdict consistent across all 20
  seeds, per (phase, input_set)? A split verdict means initialization
  variance is large enough to matter for how many seeds Stage B needs —
  worth pausing on rather than aggregating past it.
- `results[i]["pre_warmup"][input_set]["mechanism_verdict"]` — the actual
  answer to "is a random-init ALIGNN overconfident." If this comes back
  `no_initial_overconfidence`, say so plainly; don't tune sigma/LR/steps
  until an effect appears.
- Do the three input sets (`real`, `coordinate_perturbed`,
  `random_feature`) actually differ? If `random_feature` looks the same as
  `real`, that itself says something about what the encoder is doing at
  init.
- `activation_scale_summary.rms_by_call_order` — activation RMS across
  depth, BatchNorm modules flagged (`is_batchnorm`), for a single
  representative batch (not averaged across the full dataset — see the
  protocol doc for why that's the correct choice, not a shortcut).
- `STAGE_A_RAW_LOGITS_fold0.npz` — per-sample logits, keyed
  `seed{seed}__{phase}__{input_set}`, for the actual logit histograms /
  logit-vs-softmax plots (the aggregate JSON alone can't produce these).
- `head_perturbation_control` entries — always compare against this before
  concluding a real warm-up arm "did something." It isolates displacement-
  magnitude effects from the warm-up mechanism itself.

## Known limitations, as of this writing

- Never run against the real matbench dataset or an actual A100 — every
  step above is unexercised at the scale this describes. Core mechanics
  (graph construction, the real ALIGNN forward pass, hooks, warm-up
  determinism) *have* been dry-run end-to-end on a local non-A100 GPU
  against a synthetic structure, which caught and fixed two real bugs
  (see `STAGE3_STAGE_A_PROTOCOL.md`) — but that's not the same as a real
  run.
- Stage B (training-set-size sweep) and Stage C (full 5-fold run) don't
  exist yet, by design — Stage A's verdict should shape Stage B's grid.
