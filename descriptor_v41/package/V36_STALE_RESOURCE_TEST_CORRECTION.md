# v36 stale resource-test correction

DelftBlue v35 runtime pytest completed with 155 passing tests and two failures. Both failures were stale assertions in `tests/test_v13_slurm_validation.py` that required eight CPUs for every job and still identified the recovery partition as `gpu-a100`.

The v35 recovery job intentionally uses `gpu-a100-small`, for which live DelftBlue validation requires `--cpus-per-task=2`. v36 changes only those assertions. It continues to require one task for all jobs, eight CPUs for the other nine jobs, exactly two CPUs for `09_recover_calibration_f0s0.sbatch`, the correct account, valid memory, explicit time and no unnecessary nodes directive.

No executable, scientific configuration, scaler, data, split, seed, checkpoint, prediction, Slurm directive or recovery behavior changed from v35.
