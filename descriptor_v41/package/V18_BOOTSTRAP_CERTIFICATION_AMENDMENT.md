# Prospective v18 bootstrap-certification amendment

This additive operational amendment corrects the v17 circular clean-install gate before environment setup or scientific execution. It does not amend the model, data, folds, inner splits, seeds, intervention, optimizer, scheduler, epochs, batch size, CUDA dependency graph, Slurm resources, MUBen scaler, metrics, prediction schema or analysis.

The immutable archive contains two different classes of evidence:

- `DEPENDENCY_RESOLUTION_EVIDENCE.json` is passed static resolution evidence for the exact CPython 3.10.20/Linux/x86-64 target.
- `CLEAN_INSTALL_EVIDENCE.json` and `CLEAN_INSTALL_TRANSCRIPT.txt` are pending placeholders. They explicitly cannot satisfy runtime certification.

`setup_environment.sh` first verifies the static evidence, installs and tests a staging prefix in explicit bootstrap mode, clones it to the final prefix, and verifies the final prefix. It then writes the actual clean-install evidence and transcript under the final environment prefix. The finalizer validates those external files, verifies the immutable package manifest, runs the complete test suite without bootstrap mode, writes a hashed pytest log, and atomically issues the login certificate only if every step passes.

The login certificate binds the v18 package aggregate plus the absolute paths and SHA-256 hashes of the runtime evidence, transcript and pytest log. `require_certification.py` rechecks those bindings. The A100 report binds the login-certificate hash and package aggregate. Any missing, pending, mismatched or tampered artifact fails closed.

No scientific execution is authorized by static evidence, bootstrap-mode tests or the presence of a Python executable.
