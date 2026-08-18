# v18 verification report

v18 verification covers static dependency-resolution semantics, hostile pip-source isolation inherited from v15, the non-circular setup order, external runtime-evidence validation, transcript and environment-report hash binding, full-pytest-before-certificate ordering, login certificate binding, A100 certificate binding, immutable pending placeholders, package integrity, Python compilation, JSON syntax, shell syntax, Slurm resource policy, archive safety, archive permissions and clean extraction.

The exact CPython 3.10.20/Linux/glibc-2.35 clean install, runtime full pytest, DelftBlue `sbatch --test-only` and A100 runtime checks are intentionally not claimed by local static verification. They are generated and checked by the v18 setup and runbook on DelftBlue before any scientific gate.
