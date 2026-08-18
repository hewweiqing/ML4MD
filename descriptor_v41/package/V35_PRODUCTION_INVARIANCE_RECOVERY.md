# v35 production-invariance recovery

DelftBlue v34 job `10644192` passed the synthetic/validation calibration gate, performed its single authorized outer-test materialization, and then failed closed in `verify_cell.py`. The failure was caused by the production verifier retaining exact stable-`argsort` and fixed-absolute-tolerance tie-matrix equality even though the calibration contract had already adopted tolerance-aware binary-margin ordering.

v35 is additive. It preserves all v34 and earlier archives, trained checkpoints, logits, temperatures, predictions, scientific configuration and MUBen-derived scaler source.

The correction has two parts:

1. `assert_temperature_invariance` applies the declared tolerance-aware margin-ordering check in both directions. It still rejects material inversions, while treating roundoff-scale reorderings as ties. The previous quadratic pairwise tie matrix is removed.
2. When the single-use outer-test sentinel already exists, the recovery job verifies the validation fit and hashes the existing export artifacts, then proceeds directly to cell verification. It cannot call `calibrate_and_export.py` or rematerialize the dataset on that path.

No training, calibration, inference, outer-test evaluation, network, pip or Slurm operation was performed while constructing v35.
