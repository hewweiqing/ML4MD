# v29 DGL Column Correction

DelftBlue v28 coordinate-cache array 10638930 reconstructed the authorized sigma-0.040 noisy structure, atom graph and line graph successfully, then rejected every record because DGL 1.1.1 `Frame.values()` returns lazy `Column` objects. Passing those objects to `torch.isfinite` raises `TypeError`.

v29 materializes features by sorted key (`namespace[key]`) before tensor/finite validation. This is an API-compatibility correction only. It does not change sigma, structure selection, displacement seeds, perturbations, graph settings, validity thresholds, warm-up, supervised training, calibration or analysis. The coordinate protocol/seed identity remains 28 so corrected records are scientifically identical to those v28 intended to build.

If cache construction fails again, v29 atomically retains `COORDINATE_CACHE_FAILURE_DIAGNOSTIC.json` with the first 25 underlying outcome-neutral rejection exceptions. No partial shard receives a passing manifest.
