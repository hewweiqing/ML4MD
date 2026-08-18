from __future__ import annotations

import hashlib
import json
from pathlib import Path

from alignn_stage2.dependency_resolution import validate_resolution
from scripts.verify_clean_install_evidence import EXPECTED_TARGET, validate

ROOT = Path(__file__).resolve().parents[1]
LOCKS = ("BOOTSTRAP_REQUIREMENTS.txt", "PYTHON_DEPENDENCY_LOCK.txt", "TORCH_CU118_REQUIREMENT.txt",
    "DGL_CU118_REQUIREMENT.txt", "CUDA11_RUNTIME_REQUIREMENTS.txt")


def synthetic_runtime_evidence(transcript: bytes) -> dict:
    return {"schema_version": 2, "status": "passed_runtime_clean_install", "release_blocked": False,
        "target": EXPECTED_TARGET, "precertification_checks_exit_code": 0,
        "pip_check": "No broken requirements found.", "undeclared_distributions": [],
        "absent_distributions": [], "version_mismatches": {},
        "imports": {name: {"passed": True} for name in ("torch", "dgl", "alignn")},
        "sonames": {name: {"loaded": True} for name in (
            "libcudart.so.11.0", "libcublas.so.11", "libcusparse.so.11")},
        "certification_ready": True, "package_aggregate_sha256": "a" * 64,
        "static_resolution_evidence_sha256": "b" * 64,
        "environment_report_sha256": "c" * 64,
        "transcript_sha256": hashlib.sha256(transcript).hexdigest()}


def test_static_resolution_is_independently_passed_before_installation():
    evidence = json.loads((ROOT / "DEPENDENCY_RESOLUTION_EVIDENCE.json").read_text(encoding="utf-8"))
    locks = [(ROOT / name).read_text(encoding="utf-8") for name in LOCKS]
    assert evidence["status"] == "passed_static_resolution"
    assert validate_resolution(evidence, locks)["status"] == "passed"


def test_external_runtime_evidence_is_non_vacuous_and_tamper_evident():
    transcript = b"synthetic exact-runtime transcript\n"
    evidence = synthetic_runtime_evidence(transcript)
    assert validate(evidence, transcript) == []
    assert "clean-install transcript hash mismatch" in validate(evidence, transcript + b"tampered")
    assert "environment report is not bound" in validate(dict(evidence, environment_report_sha256=""), transcript)[0]
    broken = dict(evidence, certification_ready=False)
    assert "runtime evidence is not ready for final certification" in validate(broken, transcript)


def test_setup_has_non_circular_bootstrap_then_external_evidence_then_final_pytest():
    setup = (ROOT / "setup_environment.sh").read_text(encoding="utf-8")
    bootstrap = setup.index("ALIGNN_BOOTSTRAP_TEST_MODE=1")
    clone = setup.index('create --prefix "$ENV_PREFIX" --clone "$STAGING_PREFIX"')
    replay = setup.rindex("repair_bootstrap_overlay")
    certify = setup.rindex("certify_final_prefix")
    assert bootstrap < clone < replay < certify
    assert '--output "$CLEAN_EVIDENCE"' in setup and '--transcript "$CLEAN_TRANSCRIPT"' in setup
    assert '$PACKAGE_DIR/CLEAN_INSTALL_EVIDENCE.json' not in setup
    assert 'rm -f "$PROVISIONAL_MARKER"' in setup


def test_finalizer_runs_full_pytest_before_atomic_certification():
    source = (ROOT / "scripts" / "finalize_login_certification.py").read_text(encoding="utf-8")
    assert source.index("subprocess.run(command") < source.index('write_atomic(Path(args.output).resolve(), report)')
    assert 'env.pop("ALIGNN_BOOTSTRAP_TEST_MODE", None)' in source
    assert 'env["ALIGNN_CLEAN_INSTALL_EVIDENCE"]' in source
    assert '"full_pytest_status": "passed"' in source


def test_login_certification_is_bound_to_package_evidence_transcript_and_pytest():
    source = (ROOT / "scripts" / "require_certification.py").read_text(encoding="utf-8")
    for token in ("certification_workflow_version", "package_aggregate_sha256",
            "clean_install_evidence_sha256", "clean_install_transcript_sha256", "full_pytest_log_sha256"):
        assert token in source


def test_a100_certification_is_bound_to_release_login_certification():
    source = (ROOT / "scripts" / "a100_preflight.py").read_text(encoding="utf-8")
    slurm = (ROOT / "slurm" / "00_a100_preflight.sbatch").read_text(encoding="utf-8")
    assert "login_certification_sha256" in source and "package_aggregate_sha256" in source
    assert '--login-certification "$ENV_PREFIX/.alignn_stage2_v9_login_certification.json"' in slurm
    assert '--package-manifest "$SLURM_SUBMIT_DIR/PACKAGE_MANIFEST.json"' in slurm


def test_immutable_placeholder_can_only_pass_in_explicit_bootstrap_mode():
    placeholder = json.loads((ROOT / "CLEAN_INSTALL_EVIDENCE.json").read_text(encoding="utf-8"))
    assert placeholder["release_blocked"] is True
    assert placeholder["status"] == "pending_delftblue_exact_linux_validation"
    assert "outside the immutable package" in placeholder["reason"]


def test_runtime_evidence_paths_are_inside_the_final_environment_prefix():
    setup = (ROOT / "setup_environment.sh").read_text(encoding="utf-8")
    for name in ("CLEAN_EVIDENCE", "CLEAN_TRANSCRIPT", "PYTEST_LOG", "PROVISIONAL_MARKER"):
        assert f'{name}="$ENV_PREFIX/' in setup
