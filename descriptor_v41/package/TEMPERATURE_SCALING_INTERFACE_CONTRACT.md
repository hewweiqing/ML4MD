# Final temperature-scaling interface contract

The only approved implementation in v31 is `vendor/muben_temperature_scaling.py`, exact SHA-256 `108b3183400725ca53e025c6f0f92690f35069f6fd786091b9d37dc50e90f08b`, with status `approved_muben_v31_numerical_convergence_amendment`. The implementation remains MUBen-derived and uses one shared scalar; v31 changes only the declared final `log_T` gradient tolerance to `1e-7` and does not add a second calibration algorithm.

```text
fit_temperature(validation_logits, validation_labels) -> fitted object
apply_temperature(logits, fitted object) -> scaled logits
```

ALIGNN is one binary task with native logits shaped `[n,2]` and one shared scalar. Fitting minimizes unweighted inner-validation cross-entropy in float64 over `log_T`, where `T=exp(log_T)`. LBFGS strong-Wolfe fitting terminates against declared gradient/change tolerances and must report successful numerical convergence. The fitted record exposes positive finite T, objective, convergence, optimization steps, before/after validation NLL, implementation version, parameterization, dtype, tolerances, gradient and source SHA-256.

Fitting on outer-test inputs, fitting probabilities, vector/per-column temperatures, pending approval, non-positive T, non-finite values, dimension changes, raw-array mutation, prediction/ranking changes, or a source-hash mismatch are fatal. Numerically, `softmax(z/T)[:,1]` must equal `sigmoid((z[:,1]-z[:,0])/T)` within `atol=1e-7`, `rtol=1e-6`.
