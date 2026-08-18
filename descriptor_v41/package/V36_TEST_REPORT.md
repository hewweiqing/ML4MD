# v36 test report

Focused source-tree regression suite:

```text
42 passed in 0.81s
```

The suite included the complete v12 and v13 Slurm policy tests, v31-v36 recovery-policy tests, and production-workflow tests. The release builder additionally performs source and clean-extraction AST compilation, JSON validation, `bash -n`, all-job resource audit, package verification, recovery-policy behavior, production roundoff acceptance, material-inversion rejection, archive safety, normalized permission and preservation-hash checks.

No training, calibration, inference, pip, network, Slurm, cluster operation or outer-test evaluation was run during construction.
