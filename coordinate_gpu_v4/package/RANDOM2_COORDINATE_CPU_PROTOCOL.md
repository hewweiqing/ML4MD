# Random2-Coordinate CPU Protocol (v29; v28 scientific identity retained)

v28 is an implementation-complete, CPU-only successor to immutable v26 and the blocked v27 evidence release. Its scientific execution is deliberately unauthorized until a positive prospectively selected Cartesian per-axis sigma is imported and verified.

The 25 official Matbench cells are five outer folds by five seeds. Each cell trains two CPU models from a common initial ALIGNN state: Control and Random2-Coordinate. Both use the same clean supervised structures, labels, epoch sample order, 40 epochs, batch size 32, AdamW, maximum learning rate 0.001, weight decay 1e-5, OneCycleLR, float32, unweighted NLL, no class weighting, smoothing, mixed precision, accumulation, early stopping, or workers. Selection is minimum inner-validation NLL.

For Random2-Coordinate, each inner-training structure is deep-copied. Independent `N(0,sigma^2)` Cartesian displacement is added per atom and axis without removing the mean. Species, cell and PBC are preserved; only periodic fractional axes are wrapped. ALIGNN atom and line graphs are rebuilt from noisy coordinates. Clean cached graphs are never used for noisy records.

Each fold/seed coordinate cache contains exactly 3,000 valid records in deterministic atomic shards. The labels are exactly 1,500 zero and 1,500 one, deterministically shuffled, and are unrelated to true targets. A common frozen CPU ALIGNN encoder extracts pooled descriptors. Only `fc.weight` and `fc.bias` receive 938 AdamW updates at batch size 128, learning rate 1e-4 and zero weight decay. Encoder parameters and BatchNorm buffers must remain byte-identical. The supervised RNG and order are reset afterward.

Calibration fits one shared positive scalar temperature per model to native two-class validation logits `(z0,z1)` in float64 with the approved MUBen-derived module. Outer-test inputs are never used for fitting. The four reported conditions are CPU Control raw, CPU Control TS, CPU Random2-Coordinate raw, and CPU Random2-Coordinate TS.

OOF analysis concatenates the five official outer folds separately for each fixed seed. It reports NLL, positive-class Brier, fixed/adaptive/classwise ECE, accuracy, F1, ROC-AUC, mean confidence and reliability bins. The 5,000-repetition bootstrap resamples structures; every sampled structure carries all five seeds and all four conditions.
