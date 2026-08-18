# Frozen calibration metric definitions (v17)

These definitions are frozen before any outer-test analysis.

- **NLL:** sample mean of the two-class cross-entropy evaluated from the native or temperature-scaled `[z_0,z_1]` matrix.
- **ECE-15:** top-label confidence ECE. Confidence is `max(p_0,p_1)` and correctness is the predicted-class indicator. Fifteen equal-width bins span `[0,1]`; bins are `[lower,upper)` except the final bin, which includes 1.
- **Adaptive ECE-15:** top-label confidence ECE over 15 deterministic equal-mass groups. Samples are stably sorted by confidence and divided with `numpy.array_split`; ties retain sample order.
- **Classwise ECE-15:** equal-width ECE is calculated separately for class 0 using `p_0` and `1[y=0]`, and class 1 using `p_1` and `1[y=1]`. The reported classwise error is the unweighted mean of the two class errors.
- **Brier:** binary positive-class Brier score `mean((p_1-y)^2)`. It is not the two-class summed Brier score.
- **ROC-AUC:** calculated from continuous positive-class probability `p_1`, with average ranks for ties. Predicted labels are never supplied as the score.
- **Accuracy/F1:** predicted class is `argmax([z_0,z_1])`; F1 uses class 1 as positive.

Temperature invariance checks require unchanged predicted class, margin ordering, tie pattern, and ROC-AUC for every positive shared scalar temperature.

