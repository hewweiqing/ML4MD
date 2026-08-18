# Final MUBen temperature-scaling integration checklist

- [x] Complete the Uni-Mol persistent final audit: six of six cells, overall PASS.
- [x] Freeze final audit SHA-256 `f4989a94700990e768cbe595a1399f23dc79e33df2471dd4e12ec32bf5245d43`.
- [x] Implement the prospectively frozen ALIGNN extension: one shared `T=exp(log_T)`, float64 validation NLL, numerical convergence.
- [x] Freeze v31 adapter SHA-256 `108b3183400725ca53e025c6f0f92690f35069f6fd786091b9d37dc50e90f08b` and preserve the superseded v26 hash in provenance.
- [x] Reject pending/generic approval states, missing audit, wrong source hash, vector temperatures, probability fitting, and outer-test fitting.
- [x] Verify two-logit softmax versus sigmoid-margin equivalence, deterministic repeatability, positive finite T, raw immutability, prediction/ranking invariance, and extreme finite logits.
- [x] Fit Control and Random2 separately from their cached inner-validation logits.

The final adapter is bundled and approved. Reinstallation is unnecessary unless a later independently audited source is deliberately adopted as another additive package version.
