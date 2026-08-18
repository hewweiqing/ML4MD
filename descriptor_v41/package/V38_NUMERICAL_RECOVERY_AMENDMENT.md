# v38 result-blind numerical recovery amendment

This additive amendment was written after DelftBlue v37 completed all 24 remaining training cells. It uses only process states, failure types, Boolean invariance diagnostics and artifact-presence evidence. No outer-test prediction row or metric value was inspected.

## Frozen failure sets

- Pre-outer validation-only convergence diagnostics: array indices 6, 14 and 16, corresponding to cells (fold 1, seed 1), (fold 2, seed 4) and (fold 3, seed 1).
- Existing single-use export recovery: array indices 9, 10, 15, 17 and 19, corresponding to cells (1,4), (2,0), (3,0), (3,2) and (3,4).

These sets are disjoint and hard-coded. A cell outside the applicable set fails closed.

## AUROC numerical correction

Binary temperature scaling divides both native logits by one shared positive scalar. It preserves the ordering of the native margin `z_1-z_0`. AUROC is a ranking functional, and exact sigmoid probability is strictly monotone in that margin. Finite-precision sigmoid evaluation can saturate distinct large margins to the same stored probability, creating artificial ties and making exact probability-derived AUROC equality an invalid implementation gate.

v38 therefore calculates AUROC from the native binary margin, which is mathematically rank-equivalent to exact positive-class probability, and establishes raw/scaled invariance through the existing tolerance-aware native-margin ordering contract. Material margin inversions, changed labels/classes, nonfinite arrays and shape changes remain fail-closed.

## Existing-export recovery boundary

The five existing-export cells already crossed the single-use outer-test boundary and wrote complete export artifacts. v38 does not call calibration/export, inference or dataset materialization for them. It does not interpret outer-test labels and does not calculate or compare an outer-test metric. It validates only the sentinel, completion status, approved scaler hash, artifact presence and hashes, row/ID coverage, finite native-logit structure and raw/scaled margin invariance. It then creates completion provenance and atomically promotes the unchanged cell.

## Pre-outer convergence boundary

The other three cells have no outer-test sentinel. v38 repeats the already-approved MUBen fit only on their saved inner-validation logits and records permitted convergence metadata. The status remains `blocked_pending_validated_muben_numerical_resolution`. v38 does not relax the declared gradient tolerance, change optimizer behavior, substitute an algorithm or authorize outer-test access. A future authoritative MUBen-derived numerical amendment is required before those cells can proceed.
