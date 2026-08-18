from __future__ import annotations

from pathlib import Path

from scripts.validate_v19_clone_overlay import EXPECTED_ERRORS, validate

ROOT = Path(__file__).resolve().parents[1]
PREFIX = Path("/scratch/synthetic/conda_envs/alignn_matbench_is_metal_cu118_v9")


def observed_report() -> dict:
    return {"status": "failed", "environment_revision": 9,
        "environment_prefix": str(PREFIX), "expected_prefix": str(PREFIX),
        "errors": sorted(EXPECTED_ERRORS), "undeclared_distributions": [],
        "absent_distributions": [],
        "version_mismatches": {"pip": {"expected": "25.3", "actual": "26.2.1"}},
        "versions": {"setuptools": "84.0.0", "wheel": "0.47.0"},
        "imports": {name: {"passed": True} for name in ("torch", "dgl", "alignn")},
        "cuda_runtime": {"passed": True}}


def test_exact_observed_clone_overlay_is_recoverable():
    assert validate(observed_report(), PREFIX) == []


def test_any_additional_environment_problem_fails_closed():
    value = observed_report()
    value["errors"].append("unexpected scientific dependency mismatch")
    assert "failure set is not the exact observed" in validate(value, PREFIX)[0]
    value = observed_report()
    value["imports"]["alignn"]["passed"] = False
    assert any("alignn" in item for item in validate(value, PREFIX))


def test_final_clone_repairs_hash_locked_bootstrap_before_certification():
    setup = (ROOT / "setup_environment.sh").read_text(encoding="utf-8")
    clone = setup.index('create --prefix "$ENV_PREFIX" --clone "$STAGING_PREFIX"')
    repair = setup.rindex("repair_bootstrap_overlay")
    certify = setup.rindex("certify_final_prefix")
    assert clone < repair < certify
    assert 'pip --python "$ENV_PREFIX/bin/python" install --ignore-installed' in setup


def test_recovery_is_restricted_to_v19_report_and_absent_staging():
    setup = (ROOT / "setup_environment.sh").read_text(encoding="utf-8")
    assert 'V19_FAILED_REPORT="$ENV_PREFIX/.alignn_stage2_v19_environment_report.provisional.json"' in setup
    assert '! -e "$STAGING_PREFIX" && -f "$V19_FAILED_REPORT"' in setup
    assert "validate_v19_clone_overlay.py" in setup
    assert "Only the exact validated v19 post-clone" in setup
