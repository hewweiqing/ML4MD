# DelftBlue v26 package readiness

v26 additively removes the sole false-negative assertion identified by the v25 DelftBlue runtime suite (`143 passed, 1 failed`). The independent AST-based test continues to enforce distinct profiler report-path and model-output bindings. See `V26_TEST_CORRECTION.md`.

v25 additively corrects the profiler report-output variable collision observed in DelftBlue job `10633397`. The job completed its representative batch work, but the model tensor had overwritten the report path before the atomic write. `scripts/profile_100_batches.py` now keeps `output_path` and `model_output` distinct, with a regression test covering the exact failure. See `V25_PROFILE_OUTPUT_CORRECTION.md`.

The v24 login, A100, graph-cache, and non-primary smoke evidence proved the unchanged environment, GPU runtime, graph construction, frozen-file registry, and limited execution contract. Because certifications and cache provenance are fail-closed and package-identity-bound, v25 must rebind or reproduce those operational gates before profiling. No full scientific fold or seed has been trained.

v20 fixes the real v19 post-clone bootstrap-tool overlay. Conda clone preserved all scientific packages and CUDA libraries but restored its newer pip, setuptools and wheel packages over the pinned pip installations. Fresh setup now force-replays the unchanged hash-locked bootstrap requirements after cloning and before final verification.

For the existing failed v19 final prefix, setup validates the recorded report against the exact known error set, requires no staging prefix, confirms all imports and CUDA runtime passed and that no distributions are absent or undeclared, then replays only that same bootstrap lock. Any additional difference fails closed.

The v18 non-circular certification flow remains intact: static resolution proof is checked before installation; actual final-prefix evidence is generated outside the archive; full pytest runs against it; and the login certificate atomically binds the package, evidence, transcript and test-log hashes.

Fully ready: immutable package verification, hostile-pip isolation, deterministic CPython 3.10.20/CUDA 11.8 setup, static dependency proof, external runtime-evidence generation and validation, full-test certification, Slurm resource audit, test-only helper, A100 preflight binding, cache/smoke/profile/primary/grid/calibration/OOF/final-audit entry points, approved MUBen scaler, schemas and provenance.

Deliberately pending on DelftBlue: the exact Linux clean installation, generated runtime evidence, full runtime pytest, `sbatch --test-only`, A100 certification and subsequent manual scientific gates. No scientific job is authorized merely by extracting this archive.
