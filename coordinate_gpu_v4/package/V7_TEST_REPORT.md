# v7 test report

Scientific contract tests passed in both required locations:

- Working package: `29 passed`.
- Clean extraction of `alignn_stage2_delftblue_v7.tar.gz`: `29 passed`.

The suite retains the original v4 calibration/prediction tests and adds installed-source provenance, real vendored-source determinism, explicit vector-temperature rejection, extreme finite logits, and branch/fold/seed isolation checks. Covered gates include clean hash installation, wrong/missing hash rejection, one positive scalar, validation-only fitting, raw-logit immutability, label/structure-ID alignment, two-logit/margin equivalence, outer-test prohibition and fail-closed behavior.

Package-manifest verification passed in the working package and clean extraction. Bash syntax and Linux line-ending checks passed (byte-exact CRLF upstream snapshots are intentionally exempt from line-ending rewriting). The laptop does not reproduce DelftBlue's pinned CUDA 11.8 environment, so login/A100 runtime certification is recorded as not passed locally and remains a future preflight responsibility. No ALIGNN training, Slurm submission, or ALIGNN outer-test access occurred.

Authoritative machine-readable build/test details are in the adjacent `DELFTBLUE_ARCHIVE_MANIFEST_V7.json` sidecar.
