#!/bin/bash
# v28 reuses the immutable certified v26 Python prefix; it performs no installation.
set -euo pipefail
: "${ALIGNN_ENV_PREFIX:=/scratch/$USER/conda_envs/alignn_matbench_is_metal_cu118_v9}"
test -x "$ALIGNN_ENV_PREFIX/bin/python" || { echo "Missing certified v26 Python prefix: $ALIGNN_ENV_PREFIX" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
"$ALIGNN_ENV_PREFIX/bin/python" scripts/verify_cpu_environment.py
"$ALIGNN_ENV_PREFIX/bin/python" scripts/verify_package.py
echo "ALIGNN_CPU_ENVIRONMENT: PASS (reused certified prefix; no installation performed)"
