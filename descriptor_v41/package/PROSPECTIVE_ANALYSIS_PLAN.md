# Prospective analysis plan

Written before any outer-test evaluation on 2026-07-15.

## Scientific question and estimand

Does Random2-Descriptor pretraining at the pooled crystal-descriptor interface add calibration or failure-awareness benefit beyond validation-fitted temperature scaling in native ALIGNN classification of Matbench `matbench_mp_is_metal`?

The primary estimand is the paired structure-level five-fold OOF difference in NLL: `Random2-Descriptor+TS - Control+TS`. Negative is favourable. Secondary calibration estimands are Brier score, ECE-15, adaptive equal-mass ECE, and classwise calibration error. Performance and failure-awareness endpoints are reported separately.

## Fixed design

- Use the five official Matbench outer folds unchanged.
- For every fold and seed, form one deterministic stratified validation split from outer training only.
- Use identical initialization, supervised samples, validation samples, epoch orders, optimizer, scheduler, batch size, early stopping, and seeds for the paired trained branches.
- Fit scalar temperature separately on each branch's validation logits by unweighted NLL, with `T = exp(t)`.
- Never use outer-test data for training, early stopping, descriptor statistics, warm-up, temperature fitting, hyperparameter choice, or adequacy decisions.
- Train two models per fold/seed. The two temperature-scaled conditions are post-hoc derivatives.

## Gates

Before test evaluation, require a genuine ALIGNN CUDA forward/backward pass on at least two periodic structures, native reproducible logit access, exact split isolation, prospective config hashes, passing unit tests, and all specified validation adequacy checks. A failed mandatory gate stops execution without weakening the gate.

## Statistical analysis

One structure is one unit. Concatenate all five outer-test folds into one OOF vector per seed. Compute paired contrasts on aligned structures. Use 5,000 paired structure bootstrap repetitions with seed 20260715. With multiple seeds, report per-seed OOF effects and matched-seed plus paired-structure hierarchical intervals. Folds are not treated as independent scientific replicates. An interval crossing zero is not described as supported, and no equivalence claim is made without a predeclared margin.

## Interpretation

Random2-Descriptor is a latent-interface adaptation of random-label warm-up for a periodic crystal classifier; it is not raw-coordinate noise training.

