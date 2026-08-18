# v34 inherited test-count correction

The v33 DelftBlue runtime suite produced `151 passed` and one failure. The failure was the inherited v12 assertion that exactly seven regular `gpu-a100` Slurm scripts exist. The additive recovery script makes the correct count eight. v34 updates only that assertion.

No production module, scaler, approval, Slurm directive, scientific configuration, checkpoint, cache contract, dataset, split, seed or output logic changes. The v33 archive remains immutable. No training, calibration, inference, outer-test access, pip, network or Slurm operation occurred during v34 packaging.
