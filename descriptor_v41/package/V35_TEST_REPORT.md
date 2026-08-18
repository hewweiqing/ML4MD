# v35 test report

Construction-time operations did not run training, calibration, inference, pip, network, Slurm, cluster operations or outer-test evaluation.

Focused behavioral tests executed with the existing local Python 3.10 CUDA test environment:

```text
15 passed in 0.37s
```

The focused set covers production metrics/completion behavior, acceptance of roundoff-scale tie reordering, rejection of material margin inversion, MUBen hash and convergence rules, v26 cache identity, recovery ordering, sentinel policy, tolerance implementation, existing-export non-rematerialization and Slurm job-count preservation.

The release builder additionally performs Python AST compilation, JSON parsing, `bash -n`, the complete Slurm resource audit, pure recovery-policy tests, direct production-ordering behavioral checks, package-manifest verification, deterministic archive construction, clean extraction, repeated checks from the extracted tree, path safety and permission validation, and preservation-hash verification for all older archives.

The full DelftBlue runtime suite remains an operator gate because the exact certified Linux environment supplies DGL, ALIGNN, ijson and external clean-install evidence that are intentionally absent from the local packaging interpreter.
