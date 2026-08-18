# v17 verification report

Local Windows source validation completed without pip, network, Slurm, dataset access, graph construction, training, calibration, inference or outer-test evaluation.

- Python source compilation: 64 files, passed.
- JSON parsing: 23 files, passed.
- Bash/Slurm syntax: 12 files, passed.
- Static Slurm resource policy: 9 jobs, passed.
- Test inventory: 106 declared test functions across 15 files; execution pending the certified environment.
- Approved MUBen source SHA-256: preserved at `868654151b87d526f122f741183c23599913541d5d62aada5cb35c91bc510719`.
- Frozen ALIGNN configuration, all 25 split hashes, Random2 configuration, prospective amendment and MUBen approval: byte-identical to v16.

The exact CPython 3.10.20 Linux clean installation, complete pytest suite, CUDA synthetic execution contract, `sbatch --test-only`, A100 certification and non-primary ALIGNN smoke remain pending DelftBlue execution. `CLEAN_INSTALL_EVIDENCE.json` deliberately keeps `release_blocked: true`; it must not be edited manually or treated as certification.
