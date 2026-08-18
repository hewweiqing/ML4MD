# v14 local test report

The inherited suite plus v14 CUDA runtime regressions passed from the source tree: **100 passed**. The v14 tests reproduce missing-cuSPARSE and missing-search-path failures, exercise a complete synthetic dependency path and loader failure, reject old certification revisions, enforce exact CUDA component pins, verify all seven GPU scripts source the one helper before Python execution, verify the two compute scripts remain uncoupled, reject old operational prefixes, and confirm no profile approval is bundled.

Packaging additionally performs Python compilation, JSON parsing, Bash syntax, Linux newline, manifest, static Slurm resource, fake-runner test-only, archive path/duplicate/permission, immutable-v13, immutable-MUBEN, and clean-extraction checks. Pip, network, Slurm, training, calibration, inference, cluster access and outer-test evaluation are not invoked.
