# Prospective v17 scientific-integrity amendment

This amendment was made before scientific training or outer-test evaluation. It does not change the five official folds, five seeds, inner splits, model architecture, 40 epochs, batch size 32, optimizer, OneCycleLR configuration, Control/Random2 pairing, Random2 synthetic count or warm-up steps, MUBen scaler, or primary D-minus-B estimand.

## Raw-logit provenance

`native raw logits` means exactly the two columns emitted by `ALIGNN.fc` before the model's `LogSoftmax`. Every forward pass asserts that the public ALIGNN result equals `log_softmax(z)` and, when labels are present, that `NLLLoss(log_softmax(z),y)` equals `CrossEntropyLoss(z,y)`. Validation artifacts declare source `ALIGNN.fc output before LogSoftmax` and columns `[z_0,z_1]`; calibration rejects missing or different provenance.

## Random2 encoder integrity

Descriptor extraction runs with the entire model in `eval()` under `no_grad()`. Complete non-head state dictionaries, including BatchNorm running statistics and `num_batches_tracked`, must be byte-identical before and after descriptor extraction and between Control and Random2 immediately before supervised training.

## Metrics and inference

Metric definitions are frozen in `CALIBRATION_METRIC_DEFINITIONS.md`. Adaptive equal-mass ECE and macro classwise ECE are now mandatory outputs. Scaling invariance covers ranking, predicted class and ROC-AUC.

## Primary analysis and uncertainty

For every seed, the five official outer folds are concatenated before calculating `Delta_s = NLL_D,s^OOF - NLL_B,s^OOF`. The point estimate is the mean of the five seed-specific deltas. A bootstrap draw resamples a structure once and carries all of that structure's predictions across all seeds and A/B/C/D conditions. The interval is a structure-sampling CI conditional on the fixed Matbench folds and fixed seeds; seed-structure rows are not independent.

## Outcome-neutral gates

Validation adequacy checks only basic Control competence and mechanical integrity. No gate compares Random2 with Control or requires Random2 improvement. Any protocol change prompted by fold-0/seed-0 validation evidence requires a new package version and restart of all cells.

