# Coordinate-GPU-v4 status

Status recorded on 2026-08-18 from the DelftBlue execution transcript.

## FROZEN: do not edit `package/` while the 24 pending cells are in flight

`gpu_coordinate_training.py`'s resume path hard-fails on any mismatch
between a checkpoint's saved `integrity.package_aggregate_sha256` and the
current `PACKAGE_MANIFEST.json` aggregate (`checkpoint.get("integrity") !=
integrity: raise RuntimeError(...)`, `gpu_coordinate_training.py:447`).
That hash changes on **any** edit inside `package/` — not just code changes
to the training loop; a README typo fix inside `package/` would do the same
damage. If any of the 24 pending cells below are mid-training with a
`last.pt` already written, editing the package now would make its resume
hard-fail instead of continuing, losing whatever GPU-hours it had already
spent.

Whether that risk is real or hypothetical right now depends on whether any
`last.pt` checkpoint actually exists yet on DelftBlue — that can only be
checked on the cluster (this repo has no local mirror of the work root; it's
git-ignored). Until someone checks and confirms otherwise, treat `package/`
as frozen: no edits, however cosmetic, until the 24 pending cells report.
This is why the F4 per-epoch confidence/accuracy logging identified in the
Cheon & Paik figure audit (2026-08-19) was deliberately not added here —
it will land in `stage3_calibration`'s Stage B instead, once that's built,
rather than risk this package's live resume path for a change with no
benefit to the 24 cells already running under the old schema.

## Passed gates

- A100 runtime preflight.
- Corrected non-primary smoke test.
- 100-batch A100 profile, job `10657491`: `COMPLETED 0:0`.
- Profile projection: 4.602648825878898 paired 40-epoch hours.
- Peak CUDA memory reserved: 12,922,650,624 bytes (below the 35 GiB policy).
- Profile approval: approved without using scientific outcomes.
- Fold-0/seed-0 primary training, job `10657501`: `COMPLETED 0:0`.
- Primary training status: `training_complete` with `outer_test_accessed=false`.
- Primary full-grid gate: passed.

## Pending

The remaining 24 training cells and dependent calibration/export, OOF, and
final-audit jobs were submitted on DelftBlue. Their final outputs are not yet
included in this GitHub bundle. Add coordinate results only after the final
audit passes and the transferred artifacts are hash-verified.

The active cluster paths are:

```text
package:   $HOME/alignn_coordinate_gpu_v4/delftblue_coordinate_gpu_v4
work root: $HOME/ml4md/ALIGNN/gpu_coordinate_primary_v4
OOF root:  $HOME/ml4md/ALIGNN/gpu_coordinate_oof_v4
authority: $HOME/alignn_stage2_v29/delftblue_package_v29
```

The descriptor-v41 package is not a runtime dependency of this coordinate-v4
experiment.
