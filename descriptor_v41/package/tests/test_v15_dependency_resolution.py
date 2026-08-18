import copy
import json
from pathlib import Path

from alignn_stage2.dependency_resolution import validate_resolution
from alignn_stage2.cuda_runtime import REQUIRED_SONAMES

ROOT = Path(__file__).resolve().parents[1]


def evidence_and_locks():
    evidence = json.loads((ROOT / "DEPENDENCY_RESOLUTION_EVIDENCE.json").read_text(encoding="utf-8"))
    locks = [(ROOT / name).read_text(encoding="utf-8") for name in (
        "BOOTSTRAP_REQUIREMENTS.txt", "PYTHON_DEPENDENCY_LOCK.txt", "TORCH_CU118_REQUIREMENT.txt",
        "DGL_CU118_REQUIREMENT.txt", "CUDA11_RUNTIME_REQUIREMENTS.txt")]
    return evidence, locks


def test_real_clean_resolution_evidence_and_all_hash_locks_pass():
    evidence, locks = evidence_and_locks()
    result = validate_resolution(evidence, locks)
    assert result["status"] == "passed"
    assert evidence["status"] == "passed_static_resolution"
    assert evidence["release_time_tests"]["full_hash_locked_replay"].startswith("separate runtime gate")
    assert evidence["release_time_tests"]["exact_clean_install"].startswith("separate runtime gate")
    assert evidence["downloaded_wheel_count"] == 87


def test_exact_nonexistent_v14_nvjitlink_requirement_is_regressed():
    old_evidence = (ROOT / "AUDIT_REPORT.md").read_text(encoding="utf-8")
    current = (ROOT / "CUDA11_RUNTIME_REQUIREMENTS.txt").read_text(encoding="utf-8")
    assert "nvidia-nvjitlink-cu11==11.8.86" in old_evidence
    assert "nvidia-nvjitlink-cu11" not in current


def test_unavailable_locked_distribution_fails_closed():
    evidence, locks = evidence_and_locks()
    broken = copy.deepcopy(evidence)
    broken["wheels"] = [row for row in broken["wheels"] if row["normalized_name"] != "nvidia-cusparse-cu11"]
    result = validate_resolution(broken, locks)
    assert result["status"] == "failed"
    assert any("unavailable locked distributions" in error for error in result["errors"])


def test_hash_mismatch_fails_closed():
    evidence, locks = evidence_and_locks()
    corrupted = locks.copy()
    corrupted[1] = corrupted[1].replace(
        "a7a39a3bd276781e98394987d3a5701d0c4edffb633bb7a5144577f82c773598",
        "0" * 64,
    )
    result = validate_resolution(evidence, corrupted)
    assert result["status"] == "failed"
    assert any("hash mismatch" in error for error in result["errors"])


def test_setup_is_staged_atomic_and_removes_certification_on_failure():
    setup = (ROOT / "setup_environment.sh").read_text(encoding="utf-8")
    assert 'STAGING_PREFIX="${ENV_PREFIX}.staging"' in setup
    assert 'create --prefix "$ENV_PREFIX" --clone "$STAGING_PREFIX"' in setup
    assert 'rm -f "$MARKER"' in setup
    assert 'certification_marker_present":false' in setup
    assert "existing Python executable does not imply" in setup
    assert setup.index('create --prefix "$ENV_PREFIX" --clone "$STAGING_PREFIX"') < setup.rindex("certify_final_prefix")


def test_partial_prefixes_and_stale_staging_are_rejected_before_install():
    setup = (ROOT / "setup_environment.sh").read_text(encoding="utf-8")
    assert 'stale/partial staging prefix exists' in setup
    assert 'partial or uncertified v9 prefix rejected' in setup
    assert '[[ ! -x "$ENV_PREFIX/bin/python" || ! -f "$MARKER" ]]' in setup


def test_required_sonames_are_the_genuine_cuda11_graph():
    assert set(REQUIRED_SONAMES) == {"libcudart.so.11.0", "libcublas.so.11",
        "libcusparse.so.11", "libcusolver.so.11"}
    runtime = (ROOT / "alignn_stage2" / "cuda_runtime.py").read_text(encoding="utf-8")
    assert "loader(soname)" in runtime


def test_no_external_cuda12_style_or_unneeded_runtime_components():
    requirements = (ROOT / "CUDA11_RUNTIME_REQUIREMENTS.txt").read_text(encoding="utf-8").lower()
    for token in ("nvjitlink", "cudnn", "nccl", "nvrtc", "cufft", "curand", "cupti", "nvtx"):
        assert token not in requirements
