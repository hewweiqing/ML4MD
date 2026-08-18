# v12 local test report

Before archive construction, the complete inherited suite plus v12 regressions passed: **75 passed**. It is repeated from a clean archive extraction. New checks cover the DelftBlue Python 3.6.8 incompatibility, isolated Python 3.10.12 module subshell, post-subshell Slurm availability checks, deterministic Conda resolution/certified reuse, unit-normalized memory parsing, all gpu-a100 headers, six exact `8000M` full-configuration jobs and preservation of manual profile-to-primary submission.

The release builder also performs package verification, Python compilation, JSON parsing, Bash syntax checks, archive path/duplicate/debris/permission audits, v11 archive preservation and MUBen module preservation. No runtime action is part of these tests.
