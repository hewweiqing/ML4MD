#!/bin/bash
# Source this file. It is the only CUDA loader-path implementation in v20.
set -euo pipefail

_runtime_helper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_runtime_package_dir="$(cd "$_runtime_helper_dir/.." && pwd)"
: "${SCRATCH:=/scratch/$USER}"
_runtime_expected_prefix="$SCRATCH/conda_envs/alignn_matbench_is_metal_cu118_v9"
ALIGNN_ENV_PREFIX="${ALIGNN_ENV_PREFIX:-$_runtime_expected_prefix}"

_runtime_allowed_prefix="$_runtime_expected_prefix"
if [[ "${ALIGNN_RUNTIME_BOOTSTRAP_ALLOW_UNCERTIFIED:-0}" == "1" \
    && "${ALIGNN_RUNTIME_STAGING_PREFIX:-}" == "${_runtime_expected_prefix}.staging" ]]; then
  _runtime_allowed_prefix="$ALIGNN_RUNTIME_STAGING_PREFIX"
fi
if [[ "$ALIGNN_ENV_PREFIX" != "$_runtime_allowed_prefix" ]]; then
  echo "ERROR: v20 requires the certified-v9 prefix $_runtime_allowed_prefix; got $ALIGNN_ENV_PREFIX" >&2
  return 20 2>/dev/null || exit 20
fi
if [[ ! -x "$ALIGNN_ENV_PREFIX/bin/python" ]]; then
  echo "ERROR: v20 environment Python missing: $ALIGNN_ENV_PREFIX/bin/python" >&2
  return 21 2>/dev/null || exit 21
fi

_runtime_site="$ALIGNN_ENV_PREFIX/lib/python3.10/site-packages/nvidia"
_runtime_candidates=(
  "$ALIGNN_ENV_PREFIX/lib"
  "$_runtime_site/cuda_runtime/lib" "$_runtime_site/cublas/lib"
  "$_runtime_site/cusolver/lib" "$_runtime_site/cusparse/lib"
)
_runtime_joined=""
for _runtime_dir in "${_runtime_candidates[@]}"; do
  [[ -d "$_runtime_dir" ]] || continue
  case ":$_runtime_joined:" in *":$_runtime_dir:"*) continue ;; esac
  _runtime_joined="${_runtime_joined:+$_runtime_joined:}$_runtime_dir"
done
if [[ -z "$_runtime_joined" ]]; then
  echo "ERROR: no CUDA runtime directories exist under $ALIGNN_ENV_PREFIX" >&2
  return 22 2>/dev/null || exit 22
fi
export LD_LIBRARY_PATH="$_runtime_joined${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ALIGNN_ENV_PREFIX

PYTHONPATH="$_runtime_package_dir${PYTHONPATH:+:$PYTHONPATH}" \
  "$ALIGNN_ENV_PREFIX/bin/python" "$_runtime_helper_dir/verify_cuda_runtime.py" \
  --prefix "$ALIGNN_ENV_PREFIX" --library-only >/dev/null

if [[ "${ALIGNN_RUNTIME_BOOTSTRAP_ALLOW_UNCERTIFIED:-0}" != "1" ]]; then
  _runtime_cert="$ALIGNN_ENV_PREFIX/.alignn_stage2_v9_login_certification.json"
  [[ -f "$_runtime_cert" ]] || {
    echo "ERROR: v20 login certification missing: $_runtime_cert" >&2
    return 23 2>/dev/null || exit 23
  }
fi
echo "CUDA runtime: prefix=$ALIGNN_ENV_PREFIX source=certified-prefix environment_revision=9 package=stage3_calibration (reuses coordinate_gpu_v4's certified v9 environment; same version pins)"
unset _runtime_helper_dir _runtime_package_dir _runtime_expected_prefix _runtime_allowed_prefix _runtime_site
unset _runtime_candidates _runtime_joined _runtime_dir _runtime_cert
