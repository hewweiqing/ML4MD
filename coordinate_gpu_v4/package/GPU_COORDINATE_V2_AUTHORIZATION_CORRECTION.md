# GPU-coordinate v2 authorization correction

GPU-coordinate v1 correctly required the existing prospective v29 sigma authorization in its runbook, but its validator incorrectly compared that immutable artifact's origin package hash with the later GPU execution-wrapper aggregate. The exact v29 authorization therefore failed closed before cache verification.

This additive v2 release preserves v1 and all scientific settings. It accepts only the byte-identical external v29 authorization with SHA-256 `ea817b08ec5e5d150307e4a26fcbeb561b318c666a33612ee9b2b0182e624fc8`, origin package aggregate `d91ff9b17b6a75d8b38ecb62efbdbea796560426c365f8bb3a704091606183ef`, unchanged template hash, sigma 0.040 A per Cartesian axis, and `project_defined_fixed_dose` provenance. Altered artifacts remain fail-closed.

No training, calibration, inference, outer-test access, dependency installation, network operation, or Slurm operation was performed while producing this correction.
