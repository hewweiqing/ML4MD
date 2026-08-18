#!/bin/bash
set -euo pipefail

export PIP_CONFIG_FILE=/dev/null
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_FIND_LINKS PIP_NO_INDEX PIP_TRUSTED_HOST PIP_CONSTRAINT PIP_REQUIREMENT || true
export PIP_DISABLE_PIP_VERSION_CHECK=1

module load 2025
: "${SCRATCH:=/scratch/$USER}"
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${ALIGNN_ENV_PREFIX:-$SCRATCH/conda_envs/alignn_matbench_is_metal_cu118_v9}"
EXPECTED_PREFIX="$SCRATCH/conda_envs/alignn_matbench_is_metal_cu118_v9"
STAGING_PREFIX="${ENV_PREFIX}.staging"
SRC_ROOT="${ALIGNN_SRC_ROOT:-$SCRATCH/alignn_matbench_is_metal_sources}"
DATA_DIR="${ALIGNN_DATA_DIR:-$SCRATCH/alignn_matbench_is_metal_data}"
DIAG_DIR="${ALIGNN_SETUP_DIAG_DIR:-$SCRATCH/alignn_matbench_is_metal_setup_diagnostics}"
MARKER="$ENV_PREFIX/.alignn_stage2_v9_login_certification.json"
PROVISIONAL_MARKER="$ENV_PREFIX/.alignn_stage2_v22_environment_report.provisional.json"
CLEAN_EVIDENCE="$ENV_PREFIX/.alignn_stage2_v22_clean_install_evidence.json"
CLEAN_TRANSCRIPT="$ENV_PREFIX/.alignn_stage2_v22_clean_install_transcript.json"
PYTEST_LOG="$ENV_PREFIX/.alignn_stage2_v22_full_pytest.json"
V21_FAILED_EVIDENCE="$ENV_PREFIX/.alignn_stage2_v21_clean_install_evidence.json"
REPAIR_PREFIX="${ENV_PREFIX}.bootstrap_repair_v21"
BOOTSTRAP_QUARANTINE="${ENV_PREFIX}.bootstrap_overlay_quarantine_v21"
V19_FAILED_REPORT="$ENV_PREFIX/.alignn_stage2_v19_environment_report.provisional.json"
FAIL_MARKER="${ENV_PREFIX}.setup_failed.json"
LOCK_FILE="$PACKAGE_DIR/DELFBLUE_CU118_CONSTRAINTS.txt"
RUNTIME_REQUIREMENTS="$PACKAGE_DIR/CUDA11_RUNTIME_REQUIREMENTS.txt"
RUNTIME_LOCK="$PACKAGE_DIR/CUDA11_RUNTIME_LOCK.json"
PYTHON_LOCK="$PACKAGE_DIR/PYTHON_DEPENDENCY_LOCK.txt"
BOOTSTRAP_REQUIREMENTS="$PACKAGE_DIR/BOOTSTRAP_REQUIREMENTS.txt"
LOCAL_DISTRIBUTIONS="$PACKAGE_DIR/DECLARED_LOCAL_DISTRIBUTIONS.json"
RESOLUTION_EVIDENCE="$PACKAGE_DIR/DEPENDENCY_RESOLUTION_EVIDENCE.json"
TORCH_REQUIREMENT="$PACKAGE_DIR/TORCH_CU118_REQUIREMENT.txt"
DGL_REQUIREMENT="$PACKAGE_DIR/DGL_CU118_REQUIREMENT.txt"
LOCK_SHA256="$(sha256sum "$LOCK_FILE" | awk '{print $1}')"
RUNTIME_REQUIREMENTS_SHA256="$(sha256sum "$RUNTIME_REQUIREMENTS" | awk '{print $1}')"
RUNTIME_LOCK_SHA256="$(sha256sum "$RUNTIME_LOCK" | awk '{print $1}')"
PYTHON_LOCK_SHA256="$(sha256sum "$PYTHON_LOCK" | awk '{print $1}')"
BOOTSTRAP_REQUIREMENTS_SHA256="$(sha256sum "$BOOTSTRAP_REQUIREMENTS" | awk '{print $1}')"
LOCAL_DISTRIBUTIONS_SHA256="$(sha256sum "$LOCAL_DISTRIBUTIONS" | awk '{print $1}')"
RESOLUTION_EVIDENCE_SHA256="$(sha256sum "$RESOLUTION_EVIDENCE" | awk '{print $1}')"
TORCH_REQUIREMENT_SHA256="$(sha256sum "$TORCH_REQUIREMENT" | awk '{print $1}')"
DGL_REQUIREMENT_SHA256="$(sha256sum "$DGL_REQUIREMENT" | awk '{print $1}')"
ACTIVE_PREFIX="$ENV_PREFIX"
mkdir -p "$DIAG_DIR"

[[ "$ENV_PREFIX" == "$EXPECTED_PREFIX" ]] || { echo "ERROR: v21 requires the certified-v9 prefix $EXPECTED_PREFIX; got $ENV_PREFIX" >&2; exit 2; }

if [[ -n "${CONDA_EXE:-}" ]]; then
  [[ -x "$CONDA_EXE" ]] || { echo "CONDA_EXE is not executable: $CONDA_EXE" >&2; exit 2; }
  CONDA_BIN="$CONDA_EXE"
elif command -v conda >/dev/null 2>&1; then CONDA_BIN="$(command -v conda)"
elif [[ -x "$HOME/miniforge3/bin/conda" ]]; then CONDA_BIN="$HOME/miniforge3/bin/conda"
else
  echo "Conda not found. Set executable CONDA_EXE, put conda on PATH, or use $HOME/miniforge3/bin/conda." >&2
  echo "This script will not install or replace Miniforge automatically." >&2
  exit 2
fi
echo "Using Conda executable: $CONDA_BIN"

collect_failure_diagnostics() {
  local rc="$1" stamp diagnostic
  set +e
  rm -f "$MARKER"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  diagnostic="$DIAG_DIR/setup_failed_v20_${stamp}.txt"
  {
    echo "exit_code=$rc"
    echo "timestamp_utc=$stamp"
    echo "final_prefix=$ENV_PREFIX"
    echo "active_prefix=$ACTIVE_PREFIX"
    echo "staging_prefix=$STAGING_PREFIX"
    echo "certification_marker_removed=true"
    echo "IMPORTANT: an existing Python executable does not imply successful setup or certification."
    if [[ -x "$ACTIVE_PREFIX/bin/python" ]]; then
      "$ACTIVE_PREFIX/bin/python" --version
      "$ACTIVE_PREFIX/bin/python" -m pip check
      "$ACTIVE_PREFIX/bin/python" -m pip freeze --all
    else echo "active_environment_python_absent=true"; fi
  } >"$diagnostic" 2>&1
  printf '{"environment_revision":9,"status":"failed","certification_marker_present":false,"diagnostic":"%s"}\n' "$diagnostic" >"$FAIL_MARKER"
  echo "Environment setup failed closed. Diagnostic: $diagnostic" >&2
  echo "An existing Python executable is NOT certification. Remove only the failed v9 prefix/staging prefix before retry." >&2
  exit "$rc"
}
trap 'collect_failure_diagnostics $?' ERR

cert_args=(--constraints-sha256 "$LOCK_SHA256" --runtime-requirements-sha256 "$RUNTIME_REQUIREMENTS_SHA256"
  --runtime-lock-sha256 "$RUNTIME_LOCK_SHA256" --python-lock-sha256 "$PYTHON_LOCK_SHA256"
  --bootstrap-requirements-sha256 "$BOOTSTRAP_REQUIREMENTS_SHA256"
  --local-distributions-sha256 "$LOCAL_DISTRIBUTIONS_SHA256"
  --resolution-evidence-sha256 "$RESOLUTION_EVIDENCE_SHA256"
  --torch-requirement-sha256 "$TORCH_REQUIREMENT_SHA256" --dgl-requirement-sha256 "$DGL_REQUIREMENT_SHA256")

certify_final_prefix() {
  local final_python="$ENV_PREFIX/bin/python"
  PYTHONPATH="$PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$final_python" "$PACKAGE_DIR/scripts/verify_environment.py" --output "$PROVISIONAL_MARKER" \
    --expected-prefix "$ENV_PREFIX" "${cert_args[@]}"
  "$final_python" "$PACKAGE_DIR/scripts/record_clean_install_evidence.py" \
    --environment-report "$PROVISIONAL_MARKER" --package-manifest "$PACKAGE_DIR/PACKAGE_MANIFEST.json" \
    --resolution-evidence "$RESOLUTION_EVIDENCE" --output "$CLEAN_EVIDENCE" --transcript "$CLEAN_TRANSCRIPT"
  "$final_python" "$PACKAGE_DIR/scripts/verify_clean_install_evidence.py" \
    --evidence "$CLEAN_EVIDENCE" --transcript "$CLEAN_TRANSCRIPT"
  PYTHONPATH="$PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 \
    "$final_python" "$PACKAGE_DIR/scripts/finalize_login_certification.py" \
    --environment-report "$PROVISIONAL_MARKER" --evidence "$CLEAN_EVIDENCE" --transcript "$CLEAN_TRANSCRIPT" \
    --package-root "$PACKAGE_DIR" --tests "$PACKAGE_DIR/tests" --pytest-log "$PYTEST_LOG" --output "$MARKER"
  "$final_python" "$PACKAGE_DIR/scripts/require_certification.py" --kind login --path "$MARKER" --expected-prefix "$ENV_PREFIX"
  rm -f "$PROVISIONAL_MARKER"
}

repair_bootstrap_overlay() {
  [[ ! -e "$REPAIR_PREFIX" && ! -e "$BOOTSTRAP_QUARANTINE" ]] || { echo "ERROR: stale repair/quarantine path" >&2; return 4; }
  "$CONDA_BIN" create --prefix "$REPAIR_PREFIX" python=3.10.20 pip -y
  "$ENV_PREFIX/bin/python" "$PACKAGE_DIR/scripts/quarantine_bootstrap_overlay.py" --prefix "$ENV_PREFIX" --quarantine "$BOOTSTRAP_QUARANTINE"
  "$REPAIR_PREFIX/bin/python" -m pip --python "$ENV_PREFIX/bin/python" install --ignore-installed --no-deps --require-hashes \
    --index-url https://pypi.org/simple -r "$BOOTSTRAP_REQUIREMENTS"
  "$CONDA_BIN" remove --prefix "$REPAIR_PREFIX" --all -y
  test "$("$ENV_PREFIX/bin/python" -m pip --version | awk '{print $2}')" = "25.3"
}

if [[ -e "$STAGING_PREFIX" ]]; then
  echo "ERROR: stale/partial staging prefix exists: $STAGING_PREFIX" >&2
  echo "Preserve diagnostics, remove that exact staging prefix, and rerun. It will never be reused." >&2
  exit 3
fi
if [[ -e "$ENV_PREFIX" ]]; then
  PRIOR_CERT="$MARKER"
  if [[ ! -f "$PRIOR_CERT" && -f "${MARKER}.v22_preserved" ]]; then
    PRIOR_CERT="${MARKER}.v22_preserved"
  fi
  if [[ -x "$ENV_PREFIX/bin/python" && -f "$PRIOR_CERT" ]]; then
    PRIOR_OK="$("$ENV_PREFIX/bin/python" - "$PRIOR_CERT" <<'PY'
import json,sys
v=json.load(open(sys.argv[1]))
print("yes" if v.get("status")=="login_node_installation_certified" and v.get("full_pytest_status")=="passed" and v.get("package_aggregate_sha256")=="e4458e658cca66147dec413ec0715a506a72c77c6b22621f8fea69ee1e8be882" else "no")
PY
)"
    if [[ "$PRIOR_OK" == "yes" ]]; then
      if [[ "$PRIOR_CERT" == "$MARKER" ]]; then cp -p "$MARKER" "${MARKER}.v22_preserved"; fi
      rm -f "$MARKER"; ACTIVE_PREFIX="$ENV_PREFIX"; export ALIGNN_ENV_PREFIX="$ENV_PREFIX"; export ALIGNN_RUNTIME_BOOTSTRAP_ALLOW_UNCERTIFIED=1
      source "$PACKAGE_DIR/scripts/activate_cuda_runtime.sh"; unset ALIGNN_RUNTIME_BOOTSTRAP_ALLOW_UNCERTIFIED
      certify_final_prefix; rm -f "$FAIL_MARKER"; trap - ERR
      echo "Recertified unchanged environment for the v26 profiler-test correction package."; echo "A100 execution remains pending."; exit 0
    fi
  fi
  if [[ ! -x "$ENV_PREFIX/bin/python" || ! -f "$MARKER" ]]; then
    if [[ -x "$ENV_PREFIX/bin/python" && ! -f "$MARKER" && ! -e "$STAGING_PREFIX" && -f "$V21_FAILED_EVIDENCE" ]]; then
      echo "Detected the exact v21 glibc-2.28-only evidence failure; validating without modifying the repaired environment."
      "$ENV_PREFIX/bin/python" "$PACKAGE_DIR/scripts/validate_v21_glibc_only.py" --evidence "$V21_FAILED_EVIDENCE"
      ACTIVE_PREFIX="$ENV_PREFIX"; export ALIGNN_ENV_PREFIX="$ENV_PREFIX"; export ALIGNN_RUNTIME_BOOTSTRAP_ALLOW_UNCERTIFIED=1
      source "$PACKAGE_DIR/scripts/activate_cuda_runtime.sh"; unset ALIGNN_RUNTIME_BOOTSTRAP_ALLOW_UNCERTIFIED
      certify_final_prefix; rm -f "$FAIL_MARKER"; trap - ERR
      echo "Certified the repaired DelftBlue glibc-2.28 environment: $ENV_PREFIX"; echo "A100 execution remains pending."; exit 0
    elif [[ -x "$ENV_PREFIX/bin/python" && ! -f "$MARKER" && ! -e "$STAGING_PREFIX" && -f "$V19_FAILED_REPORT" ]]; then
      echo "Detected the exact v19 post-clone bootstrap-tool overlay; validating before narrow recovery."
      "$ENV_PREFIX/bin/python" "$PACKAGE_DIR/scripts/validate_v19_clone_overlay.py" \
        --report "$V19_FAILED_REPORT" --expected-prefix "$ENV_PREFIX"
      ACTIVE_PREFIX="$ENV_PREFIX"
      repair_bootstrap_overlay
      export ALIGNN_ENV_PREFIX="$ENV_PREFIX"
      export ALIGNN_RUNTIME_BOOTSTRAP_ALLOW_UNCERTIFIED=1
      source "$PACKAGE_DIR/scripts/activate_cuda_runtime.sh"
      unset ALIGNN_RUNTIME_BOOTSTRAP_ALLOW_UNCERTIFIED
      certify_final_prefix
      rm -f "$FAIL_MARKER"
      trap - ERR
      echo "Recovered and certified the exact v19 post-clone environment: $ENV_PREFIX"
      echo "A100 execution remains pending. Submit only slurm/00_a100_preflight.sbatch next."
      exit 0
    else
      echo "ERROR: partial or uncertified v9 prefix rejected: $ENV_PREFIX" >&2
      echo "Only the exact validated v19 post-clone bootstrap-tool overlay is recoverable." >&2
      exit 3
    fi
  fi
  PYTHONPATH="$PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}" "$ENV_PREFIX/bin/python" "$PACKAGE_DIR/scripts/verify_dependency_resolution.py"
  export ALIGNN_ENV_PREFIX="$ENV_PREFIX"
  source "$PACKAGE_DIR/scripts/activate_cuda_runtime.sh"
  "$ENV_PREFIX/bin/python" "$PACKAGE_DIR/scripts/require_certification.py" --kind login --path "$MARKER" --expected-prefix "$ENV_PREFIX"
  "$ENV_PREFIX/bin/python" "$PACKAGE_DIR/scripts/verify_clean_install_evidence.py" \
    --evidence "$CLEAN_EVIDENCE" --transcript "$CLEAN_TRANSCRIPT"
  PYTHONPATH="$PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$ENV_PREFIX/bin/python" "$PACKAGE_DIR/scripts/verify_environment.py" --check-only --expected-prefix "$ENV_PREFIX" "${cert_args[@]}"
  trap - ERR
  echo "Reused fully certified revision-9 environment: $ENV_PREFIX"
  exit 0
fi

rm -f "$FAIL_MARKER"
ACTIVE_PREFIX="$STAGING_PREFIX"
"$CONDA_BIN" create --prefix "$STAGING_PREFIX" python=3.10.20 pip -y
PY="$STAGING_PREFIX/bin/python"
"$PY" -m pip install --no-deps --require-hashes --index-url https://pypi.org/simple -r "$BOOTSTRAP_REQUIREMENTS"
test "$("$PY" -m pip --version | awk '{print $2}')" = "25.3"
PYTHONPATH="$PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$PACKAGE_DIR/scripts/verify_dependency_resolution.py"

mkdir -p "$SRC_ROOT" "$DATA_DIR"
if [[ ! -d "$SRC_ROOT/alignn/.git" ]]; then git clone https://github.com/usnistgov/alignn.git "$SRC_ROOT/alignn"; fi
git -C "$SRC_ROOT/alignn" fetch --all --tags
git -C "$SRC_ROOT/alignn" checkout --detach f2366daa3413d28a825b46e34d001b5549b05a40
test "$(git -C "$SRC_ROOT/alignn" rev-parse HEAD)" = f2366daa3413d28a825b46e34d001b5549b05a40
if [[ ! -d "$SRC_ROOT/matbench/.git" ]]; then git clone https://github.com/materialsproject/matbench.git "$SRC_ROOT/matbench"; fi
git -C "$SRC_ROOT/matbench" fetch --all --tags
git -C "$SRC_ROOT/matbench" checkout --detach 936176db18ca4cd7b38cbd957c017a5bac770c6b
test "$(git -C "$SRC_ROOT/matbench" rev-parse HEAD)" = 936176db18ca4cd7b38cbd957c017a5bac770c6b

"$PY" -m pip install --no-deps --require-hashes --index-url https://download.pytorch.org/whl/cu118 -r "$TORCH_REQUIREMENT"
"$PY" -m pip install --no-deps --require-hashes --no-index -r "$DGL_REQUIREMENT"
"$PY" -m pip install --no-deps --require-hashes --index-url https://pypi.org/simple -r "$RUNTIME_REQUIREMENTS"
"$PY" -m pip install --no-deps --require-hashes --index-url https://pypi.org/simple -r "$PYTHON_LOCK"
"$PY" -m pip install --no-deps "$SRC_ROOT/alignn"
"$PY" -m pip install --no-deps "$SRC_ROOT/matbench"

export ALIGNN_ENV_PREFIX="$STAGING_PREFIX"
export ALIGNN_RUNTIME_BOOTSTRAP_ALLOW_UNCERTIFIED=1
export ALIGNN_RUNTIME_STAGING_PREFIX="$STAGING_PREFIX"
source "$PACKAGE_DIR/scripts/activate_cuda_runtime.sh"
unset ALIGNN_RUNTIME_STAGING_PREFIX
"$PY" -m pip check
PYTHONPATH="$PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" "$PACKAGE_DIR/scripts/verify_environment.py" --check-only --allow-staging \
  --expected-prefix "$STAGING_PREFIX" "${cert_args[@]}"
PYTHONPATH="$PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$PACKAGE_DIR/scripts/verify_package.py"
ALIGNN_BOOTSTRAP_TEST_MODE=1 PYTHONPATH="$PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest --assert=plain -q -p no:cacheprovider "$PACKAGE_DIR/tests"

DATASET="$DATA_DIR/matbench_mp_is_metal.json.gz"
if [[ ! -f "$DATASET" ]]; then curl --fail --location 'https://ml.materialsproject.org/projects/matbench_mp_is_metal.json.gz' --output "$DATASET"; fi
echo "9a028ed5750a4c76ca36e9f3c8d48fe0bf3fb21b76ec2289e58ae7048d527919  $DATASET" | sha256sum --check

unset ALIGNN_RUNTIME_BOOTSTRAP_ALLOW_UNCERTIFIED
"$CONDA_BIN" create --prefix "$ENV_PREFIX" --clone "$STAGING_PREFIX" -y
ACTIVE_PREFIX="$ENV_PREFIX"
PY="$ENV_PREFIX/bin/python"
export ALIGNN_ENV_PREFIX="$ENV_PREFIX"
export ALIGNN_RUNTIME_BOOTSTRAP_ALLOW_UNCERTIFIED=1
source "$PACKAGE_DIR/scripts/activate_cuda_runtime.sh"
unset ALIGNN_RUNTIME_BOOTSTRAP_ALLOW_UNCERTIFIED
"$CONDA_BIN" remove --prefix "$STAGING_PREFIX" --all -y
repair_bootstrap_overlay
certify_final_prefix
rm -f "$FAIL_MARKER"
trap - ERR
echo "Login-node installation certified and dependency-consistent: $ENV_PREFIX"
echo "A100 execution remains pending. Submit only slurm/00_a100_preflight.sbatch next."
