# v20 Conda-clone bootstrap overlay correction

The real v19 staging environment passed 138 tests and contained the exact pinned bootstrap tools. After the prefix-aware Conda clone, the final report showed that Conda package relinking had replaced only those tools: pip 26.2.1 instead of 25.3, setuptools 84.0.0 instead of 80.9.0 and wheel 0.47.0 instead of 0.45.1. No distributions were missing or undeclared; torch, DGL and ALIGNN imports and every CUDA 11 library passed.

v20 retains the prefix-aware clone and then force-replays `BOOTSTRAP_REQUIREMENTS.txt` with `--no-deps --require-hashes` and the explicit PyPI index before final verification. This restores the frozen versions and prevents Conda clone mechanics from determining the certified Python bootstrap tools.

The recovery path for the existing v19 final prefix is deliberately narrow. It accepts only the exact v19 provisional report, exact three-error set, exact tool versions, empty absent/undeclared sets, passed CUDA runtime and passed torch/DGL/ALIGNN imports. It refuses all other partial prefixes. After replay it regenerates v20 evidence, reruns full pytest and binds the v20 aggregate into a new login certificate.

No dependency pin, model, dataset, split, seed, intervention, temperature scaler, training, analysis or Slurm resource is changed.
