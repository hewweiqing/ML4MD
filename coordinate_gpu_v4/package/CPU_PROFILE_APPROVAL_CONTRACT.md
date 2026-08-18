# CPU Profile Approval Contract

The non-primary profile uses fold 0/seed 0 inner-training data only, verifies the authorized coordinate cache, performs the complete coordinate head warm-up, and measures 100 full frozen-configuration CPU optimizer batches. It records timing, RSS, projected paired 40-epoch time, package/config/cache/authorization hashes, and confirms no outer-test access.

Profiling does not approve primary training. A human must inspect `preflight/CPU_COORDINATE_100_BATCH_PROFILE.json` and run `scripts/review_cpu_profile.py --approve`. Primary scripts verify that the approval hashes the current profile and current sigma authority. A projection over 24 hours is not hidden: execution remains resumable and the reviewer explicitly acknowledges the 24-hour segmentation.
