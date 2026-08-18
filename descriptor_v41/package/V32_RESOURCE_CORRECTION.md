# v32 DelftBlue resource correction

Real DelftBlue `sbatch --test-only` validation of immutable v31 rejected `slurm/09_recover_calibration_f0s0.sbatch` because `gpu-a100-small` permits at most two CPUs per task while the recovery job requested eight.

v32 changes only that recovery job's partition from `gpu-a100-small` to `gpu-a100`. It preserves `--ntasks=1`, `--cpus-per-task=8`, `--gpus-per-task=1`, `--mem-per-cpu=8000M`, the three-hour wall time, account, commands, paths and output directives. Regular `gpu-a100` is appropriate because the recovery includes outer-split graph loading and inference, not only scalar fitting, and retains the already profiled host-memory envelope.

The v31 temperature-scaling amendment, scaler hash, v26 cache identity, trained artifacts, scientific configuration and recovery access boundaries are unchanged. No training, calibration, inference, outer-test access, network, pip or Slurm operation occurred while creating v32.
