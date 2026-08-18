# v33 binary-margin ordering correction

DelftBlue v32 recovery job `10643036` passed A100 certification and completed both inner-validation temperature fits, then failed closed before writing its validation report or outer-test sentinel. The adapter compared the complete stable `argsort` permutations of raw and scaled binary margins. IEEE-754 division and subsequent subtraction can change the ordering within numerically indistinguishable tie groups even though division by one positive scalar cannot materially invert the mathematical ordering.

v33 replaces only that exact-permutation assertion with a tolerance-aware monotonicity check. Raw margins are stably sorted once; scaled margins in that order must be nondecreasing within the already declared contract tolerance `atol=1e-7`, `rtol=1e-6`. Any inversion exceeding that allowance still fails closed. This check is linearithmic rather than pairwise-quadratic and therefore remains suitable for full validation and outer-test vectors.

The MUBen-derived module and its SHA-256 `108b3183400725ca53e025c6f0f92690f35069f6fd786091b9d37dc50e90f08b` are unchanged. The fitted temperatures, objective, convergence rule, one-task/one-shared-scalar semantics, native logits, model, checkpoints, data, splits, cache, seeds and Slurm resources are unchanged. Job `10643036` created no outer-test sentinel and accessed no outer-test outcome.
