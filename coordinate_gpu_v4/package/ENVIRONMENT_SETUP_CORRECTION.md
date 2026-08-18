# v16 environment setup correction

v15 correctly failed closed on DelftBlue: its Python 3.10.20 marker environment activated pytest's `exceptiongroup>=1` dependency, which was absent from the hash lock. No v8 certification was written.

v16 uses only `/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9` and its `.staging` sibling. It pins pip to 25.3 before any production dependency stage. Every enumerated remote distribution is version- and SHA-256-pinned and installed with `--no-deps --require-hashes`; the complete graph is then checked with `pip check`. Exact installed versions and the absence of undeclared distributions are certification requirements. Any failure removes certification and writes the adjacent failure marker.
