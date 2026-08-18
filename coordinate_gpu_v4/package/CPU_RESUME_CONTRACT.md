# Deterministic CPU Resume Contract

Training checkpoints atomically preserve model, optimizer, OneCycleLR scheduler, epoch, next batch position, explicit sample order, Python RNG, NumPy RNG, torch CPU RNG, best validation NLL/checkpoint, history, optimizer-step count, and package/config/clean-cache/coordinate-cache/authorization hashes.

Slurm sends `SIGUSR1` five minutes before wall time. At the next completed optimizer batch the process atomically writes `last.pt` and `INCOMPLETE_RESUME_REQUIRED.json`, then returns 75. A partial run never writes `COMPLETE.json`. Resubmitting the same cell with unchanged paths and authority resumes from the exact next batch. `COMPLETE.json` is created only after both branches, validation-logit artifacts, approved validation-only calibration and prediction export finish.

Resume is guaranteed for the same certified CPU software and architecture. A hash mismatch fails closed; never delete or edit a checkpoint to force continuation.
