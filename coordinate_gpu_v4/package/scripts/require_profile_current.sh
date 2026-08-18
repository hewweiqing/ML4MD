#!/bin/bash
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: require_profile_current.sh <python> <cache-root>" >&2
  exit 64
fi
PYTHON_EXECUTABLE="$1"
CACHE_ROOT="$2"
exec "$PYTHON_EXECUTABLE" scripts/require_profile.py \
  --profile preflight/FULL_CONFIG_100_BATCH_PROFILE.json \
  --approval preflight/PROFILE_APPROVAL.json \
  --certification preflight/A100_RUNTIME_CERTIFICATION.json \
  --package-manifest PACKAGE_MANIFEST.json \
  --execution-plan STAGE_2_EXECUTION_PLAN.json \
  --cache-manifest "$CACHE_ROOT/STRUCTURE_CACHE_MANIFEST.json" \
  --resource-policy RESOURCE_POLICY.json \
  --primary-slurm slurm/08_primary_fold0_seed0.sbatch
