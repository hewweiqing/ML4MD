# Resource profile and approval contract

`FULL_CONFIG_100_BATCH_PROFILE.json` is non-primary runtime evidence. It must be produced on the certified A100 after the exact schema-v2 structure cache has passed. `PROFILE_APPROVAL.json` is not distributed.

Approval requires an identified reviewer, UTC time, decision and reasons. It binds SHA-256 identities for the profile, A100 certification, package aggregate, execution plan, cache manifest, resource policy and primary Slurm script. The conservative paired upper bound must use the declared safety factor and consume at most 80% of primary wall time; projected host memory, peak CUDA reserved memory and projected full-grid disk must each consume at most 80% of their verified capacity.

An unavailable, missing, non-finite, contradictory or changed input prohibits approval. A rejected, copied, stale or mismatched record blocks primary, grid, calibration, OOF and final-audit jobs. Resource or identity changes require a new profile and approval; packaged-file changes require a new additive release.
