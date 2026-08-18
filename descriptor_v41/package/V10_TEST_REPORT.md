# v10 local test report

Before archive construction, the complete source suite passed: **47 passed**.

Coverage includes final-status/hash approval, positive `log_T`, float64 convergence, deterministic fitting, extreme logits, two-class softmax/sigmoid identity, shared-scalar enforcement, outer-test-fit rejection, all 25 split hashes, exact array mapping, staged 100-batch/primary gates, structure-shard boundaries and duplicate-location rejection, global-cache reuse in calibration, atomic promotion, hash-verified resume, complete metric/reliability output, failed-dependency recovery commands, pip isolation, and archive permission policy.

Python compile checks, JSON parsing, and Git Bash `bash -n` for setup plus every Slurm script passed. The release builder must repeat all 47 tests after clean extraction and execute the exact Linux runbook verification command. Local tests use Python 3.12 with isolated NumPy 1.26.4, pytest 8.4.2, and CPU PyTorch 2.2.0; DelftBlue remains frozen at Python 3.10/CUDA 11.8.

No scientific training, Slurm submission, cluster operation, model test prediction, or test-fold evaluation was performed.
