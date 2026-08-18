# Cross-Experiment Comparison Policy

The primary v28 comparison is paired CPU Random2-Coordinate versus paired CPU Control within the same fold, seed, data order, dependency environment and CPU execution path. v26 GPU Control is not the primary comparator because that would confound intervention and training device.

Any later v26-versus-v28 comparison is explicitly secondary and descriptive. It must retain device, package aggregate, checkpoint and sigma-authorization provenance and must not replace the paired v28 estimand.
