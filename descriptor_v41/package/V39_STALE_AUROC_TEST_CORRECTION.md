# v39 stale AUROC-definition test correction

DelftBlue v38 runtime pytest produced 165 passes and one failure. The failing assertion still required the superseded text `continuous positive-class probability` even though v38 deliberately and prospectively changed the numerical AUROC implementation to use the native binary margin, which is rank-equivalent to exact positive-class probability and avoids finite-precision sigmoid saturation.

v39 changes only that assertion to require the v38 definition prefix `native binary logit margin`. It does not change production code, the MUBen scaler, calibration, model, data, splits, seeds, checkpoints, predictions, Slurm resources, recovery allowlists or outer-test policy. v38 remains immutable.
