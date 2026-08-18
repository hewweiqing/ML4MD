# v17 implementation report

Implemented prospectively from immutable v16:

- executable pre-LogSoftmax `ALIGNN.fc` provenance and NLL/CrossEntropy equivalence checks;
- singleton-batch normalization for native ALIGNN squeeze behavior;
- complete encoder state and BatchNorm-buffer immutability checks;
- sample-weighted loss provenance, exact OneCycle optimizer-step checks, and deterministic resume state for Python, NumPy, torch CPU, CUDA and manual sample-order generation;
- fixed ECE/Brier/AUC definitions plus adaptive and classwise calibration errors;
- structure-clustered bootstrap carrying all fixed seeds and conditions;
- explicitly outcome-neutral validation gates;
- a synthetic execution-contract check included in the non-primary DelftBlue smoke job.

No scientific training, calibration, inference, graph construction, Slurm submission or outer-test evaluation was performed while preparing v17.

