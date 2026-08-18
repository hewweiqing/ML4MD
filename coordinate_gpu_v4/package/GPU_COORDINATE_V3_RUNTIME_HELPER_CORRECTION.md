# GPU-coordinate v3 runtime-helper correction

GPU-coordinate v2 passed source and Slurm simulation checks but omitted `scripts/verify_cuda_runtime.py`, which its certified CUDA activation script invokes inside every GPU job. DelftBlue preflight job 10653898 therefore failed before model construction, data access, or scientific execution.

This additive v3 release restores the byte-identical verifier from the preserved v26 certified runtime lineage (SHA-256 `656c167f228562f3e1e70069ee15853e8025f301586402b3c583191e21b968f6`) and adds a behavioral release test requiring both the helper and activation reference. All v2 authorization-lineage corrections and frozen scientific settings are preserved.

No training, calibration, inference, outer-test access, dependency installation, network operation, or Slurm operation was performed while producing v3.
