# v11 local test report

Before release construction, the complete source suite passed: **62 passed**. It covers all inherited MUBen, split, paired-initialization, deterministic-resume, promotion, prediction, OOF and final-audit behavior plus v11 cache/profile controls.

New behavioral coverage includes complete cache provenance, stale dataset/settings/dependency/release rejection, atom/line/metadata corruption quarantine and rebuild, interrupted triplet recovery, exhaustive graph/count/hash verification, phase-boundary truthfulness, paired runtime arithmetic, host/GPU/disk requirements, exact approval bindings, stale approval rejection, CLI approval/require behavior and absence of automatic profile-to-primary submission.

The release builder repeats source and clean-extraction tests, the exact runbook package-verification command, Python compilation, JSON parsing, Bash syntax, manifest verification, archive safety and normalized permission checks. Runtime training, pip, network, Slurm, cluster and outer-test operations are outside this report and remain pending.
