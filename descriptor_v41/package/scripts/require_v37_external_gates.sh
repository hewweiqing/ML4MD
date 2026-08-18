#!/bin/bash
set -euo pipefail
if [ "$#" -ne 5 ]; then
  echo "usage: $0 ENV_PREFIX V26_ROOT V36_ROOT CACHE_ROOT FINAL_ROOT" >&2
  exit 2
fi
ENV_PREFIX="$1"; V26_ROOT="$2"; V36_ROOT="$3"; CACHE_ROOT="$4"; FINAL_ROOT="$5"
V37_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
PY="$ENV_PREFIX/bin/python"
PYTHONPATH="$V26_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" "$V26_ROOT/scripts/require_certification.py" \
    --kind a100 \
    --path "$V26_ROOT/preflight/A100_RUNTIME_CERTIFICATION.json" \
    --expected-prefix "$ENV_PREFIX"
(
  cd "$V26_ROOT"
  PYTHONPATH="$V26_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    bash scripts/require_profile_current.sh "$PY" "$CACHE_ROOT"
)
PYTHONPATH="$V36_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" "$V36_ROOT/scripts/require_primary_gate.py" \
    --path "$V36_ROOT/preflight/PRIMARY_FOLD0_GATE_PASSED.json"
PYTHONPATH="$V37_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" "$V37_ROOT/scripts/validate_v37_full_grid_inputs.py" \
    --v26-root "$V26_ROOT" --v36-root "$V36_ROOT" \
    --cache-root "$CACHE_ROOT" --final-root "$FINAL_ROOT"
echo "V37_EXTERNAL_FULL_GRID_GATES: PASS"
