# v40 post-training, pre-outer-test numerical-resolution amendment

## Timing and scientific status

This additive decision was authorized on 2026-08-16 after training and after inspection of inner-validation optimizer diagnostics, but before any outer-test access for the three affected cells. It was not part of the original prespecified protocol and is not described as such.

No outer-test prediction, label, probability, metric or comparison informed this decision. The fixed affected set is `(fold 1, seed 1)`, `(fold 2, seed 4)` and `(fold 3, seed 1)`.

## Authorized numerical-resolution rule

The existing authoritative MUBen-derived implementation and source SHA-256 remain unchanged. A fit is numerically resolved when all of the following hold:

- the absolute gradient of validation NLL with respect to `log_T` is at most `1e-6`;
- the single shared scalar temperature is finite and strictly positive;
- validation NLL is finite and does not increase;
- two independent invocations on the immutable inner-validation artifact return exactly identical public fit metadata and loss curve;
- the approved MUBen source SHA-256 is `108b3183400725ca53e025c6f0f92690f35069f6fd786091b9d37dc50e90f08b`;
- the fit scope is inner validation only.

The implementation's original `converged` Boolean and `convergence_status` are preserved. A nonconverged fit is never relabeled as converged. Instead, provenance records the separate fact `numerical_resolution_authorized: true` and binds it to the generated resolution-record SHA-256.

## Fail-closed boundary

Only the three exact cells and six exact validation-artifact hashes in `V40_NUMERICAL_RESOLUTION_AUTHORIZATION.json` are eligible. The validator refuses a cell with an outer-test sentinel, a changed artifact, a changed MUBen source, nondeterminism, nonfinite values, NLL increase or gradient above `1e-6`.

The validation-only resolution record must exist and pass before calibration/export can cross the outer-test boundary. Training is not repeated. OOF consolidation remains prohibited until all 25 cells are complete.
