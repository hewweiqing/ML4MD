# v31 numerical-convergence amendment

## Trigger and access boundary

DelftBlue job `10641918` completed Control and Random2-Descriptor training for fold 0, seed 0, then failed closed while fitting temperatures from inner-validation native logits. `OUTER_TEST_ACCESS_STARTED.json` was absent. No outer-test ID, label, prediction or metric was accessed.

The v26 implementation made one PyTorch strong-Wolfe LBFGS call with a declared 500-iteration limit but recorded five closure evaluations. It returned finite positive temperatures and lower validation NLLs, yet its post-fit convergence test used an effective gradient threshold of `1e-8`, which neither branch met. Control ended at `6.19290274215777e-8`; Random2 ended at `1.622938891964708e-8`.

## Amendment

v31 preserves the MUBen-derived algorithm and changes only the declared final `log_T` gradient tolerance to `1e-7`. Convergence requires the actual final absolute `log_T` gradient to be at most `1e-7`, a finite positive scalar temperature, and non-increasing validation NLL within the unchanged `1e-12` comparison allowance. The previous undocumented `tolerance_grad * 10` post-fit multiplier is removed.

Unchanged elements are one binary task, one shared scalar temperature, `T=exp(log_T)`, native `[n,2]` raw logits, float64 cross-entropy, inner-validation-only fitting, strong-Wolfe LBFGS, initial `T=1`, and a maximum of 500 optimizer iterations. No independent calibration algorithm is introduced.

## Provenance and limitations

The decision used numerical diagnostics from the two inner-validation fits after the fail-closed event. It did not use outer-test outcomes or Control-versus-Random2 superiority. It is a transparently recorded post-training, pre-outer-test numerical amendment and is not claimed to have been frozen before training.

The v26 module SHA-256 `868654151b87d526f122f741183c23599913541d5d62aada5cb35c91bc510719` remains immutable. The v31 module has its own approval status and SHA-256. Any old, pending, generic, missing or mismatched approval fails closed.
