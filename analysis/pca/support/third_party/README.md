# Upstream attribution

`anchor_selection.py` is an unmodified copy of WMD-group/PlatonicRep at commit
`56cba2354f0879c536dac68d70d509fc82a2c6ae`, copyright Zhenzhu Li.
Its SHA-256 is `f6ace5b202b896597d1e80de999acc5d15cf4411a93dfc57341548b98e91499b`.
The upstream MIT notice is retained in `LICENSE.PlatonicRep`.

Source: https://github.com/WMD-group/PlatonicRep/blob/56cba2354f0879c536dac68d70d509fc82a2c6ae/anchor_selection.py

Only `DIRECTSelector` is called, with MAML 2025.4.3, 100 requested anchors,
threshold_init=0.3 and k_per_cluster=1. No wrapper code is changed or extracted
into a replacement algorithm. Its optional plotting/other selectors are unused.
The external MAML dependency retains its own BSD-3-Clause licence.
