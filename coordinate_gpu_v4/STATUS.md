# Coordinate-GPU-v4 status

Status recorded on 2026-08-18 from the DelftBlue execution transcript.

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
