# DelftBlue package v26 audit report

## v26 additive audit

The v25 runtime suite passed 143 tests and failed only because a forbidden substring was also contained in the correct identifier `model_output`. v26 removes that logically invalid assertion while retaining the AST binding test and all positive profiler assertions. Scientific execution remains blocked pending successful v26 gates and profile approval.

## v25 additive audit

DelftBlue job `10633397` failed closed after representative profiling because the local name `output` referred first to the requested report path and later to a model tensor. v25 renames these values `output_path` and `model_output`, respectively. The calculation, measured phases, batch count, model, optimizer, scheduler, graph cache, Random2 intervention, resources, and approval policy are unchanged. A regression test rejects reintroduction of the colliding assignment. No profile artifact was produced by the failed job, no primary training was launched, and no outer-test outcome was accessed.

## v20 observed clone-overlay correction

The v19 final-prefix report proved the prefix-aware clone overlaid only pip, setuptools and wheel after a passed staging installation. v20 restores the frozen versions from the existing hash-locked bootstrap file after every clone and provides an exact-report-gated recovery for that final prefix. The scientific environment and CUDA graph were already correct and are not rebuilt by recovery.

## v19 observed-failure correction

The real v18 staging environment successfully installed ALIGNN and Matbench, loaded the CUDA runtime and passed `pip check`; it failed only because `verify_environment.py` could not import the package-local `alignn_stage2` module. v19 adds explicit package-root propagation at all three setup verifier calls and self-bootstrapping import discovery inside the verifier. No dependency, scientific, Slurm-resource or calibration behavior changed.

## v18 operational correction

v18 is additive from immutable v17. v17 archive SHA-256 `fa3292f58a52f4230e40d0c9ccd5d4743c055a918f3725f754ebb07d8e1722ef` and package aggregate `8728aa55dd6cda09b1d4417dfe2a231bf7318c6c582b5d0308aeb4399919199a` are preserved. The correction removes a circular gate in which setup required a passed clean-install artifact before setup could create that artifact. Static dependency-resolution evidence now has the distinct status `passed_static_resolution`. Runtime clean-install evidence, its transcript and the complete pytest log are generated outside the immutable extraction under the final environment prefix and hash-bound into the atomic login certificate. The A100 certificate is then bound to that login certificate and the v18 package aggregate.

## Scientific-integrity correction scope

The inherited v17 implementation makes pre-LogSoftmax `ALIGNN.fc` logit provenance executable, proves descriptor extraction cannot mutate BatchNorm state, freezes all calibration metrics, uses one structure resample carrying all fixed seeds and A/B/C/D conditions, and includes synthetic execution checks in the smoke gate. v18 changes none of those behaviors. Frozen training hyperparameters, data partitions and the approved MUBen module are unchanged.

## Preserved environment correction

The immutable v15 archive was verified at SHA-256 `5c85d7fc7e104fc2e69014b5cad5d1a9b2bed92d2aab8b9c70ab443933aa66f8` and aggregate `af3f32659354aebf2c4618b067c82f249b01639eddcde7fff769a068651b0a53`. The inherited environment evaluates every recorded dependency under CPython 3.10.20/Linux/x86-64 semantics. Its 83-distribution closure and hash-locked, no-dependency transactions remain unchanged.

## Preservation

The CUDA graph remains runtime 11.8.89, cuBLAS 11.11.3.6, cuSPARSE 11.7.5.86, and cuSOLVER 11.4.1.48. The invalid historical requirement `nvidia-nvjitlink-cu11==11.8.86` remains absent. v31 pins the MUBen-derived scaler at SHA-256 `108b3183400725ca53e025c6f0f92690f35069f6fd786091b9d37dc50e90f08b`; the superseded v26 scaler remains preserved at `868654151b87d526f122f741183c23599913541d5d62aada5cb35c91bc510719`. The model architecture, data, splits, seeds, optimizer, scheduler, epoch count, batch size and Random2 warm-up are unchanged.

No pip, network, training, calibration, inference, Slurm operation, graph caching or outer-test evaluation was performed while preparing v20. Final DelftBlue certification and real A100 certification remain pending; scientific execution remains prohibited until their certificates pass.
