from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement

from alignn_stage2.dependency_resolution import (
    normalize, parse_locks, target_environment, validate_resolution,
)
from scripts.verify_clean_install_evidence import resolve_paths, validate as validate_clean_install

ROOT = Path(__file__).resolve().parents[1]
LOCK_NAMES = (
    "BOOTSTRAP_REQUIREMENTS.txt", "PYTHON_DEPENDENCY_LOCK.txt",
    "TORCH_CU118_REQUIREMENT.txt", "DGL_CU118_REQUIREMENT.txt",
    "CUDA11_RUNTIME_REQUIREMENTS.txt",
)


def evidence_and_locks():
    evidence = json.loads((ROOT / "DEPENDENCY_RESOLUTION_EVIDENCE.json").read_text(encoding="utf-8"))
    locks = [(ROOT / name).read_text(encoding="utf-8") for name in LOCK_NAMES]
    return evidence, locks


def test_every_requirement_is_exactly_pinned_and_hashed():
    _, locks = evidence_and_locks()
    parsed, errors = parse_locks(locks)
    assert not errors
    assert parsed
    assert all(version == "direct-url" or version for version, _ in parsed.values())
    assert all(hashes and all(len(item) == 64 for item in hashes) for _, hashes in parsed.values())


def test_exact_v15_python310_marker_failure_is_reproduced_and_closed():
    evidence, locks = evidence_and_locks()
    v15 = copy.deepcopy(evidence)
    v15["status"] = "passed_static_resolution"
    v15["wheels"] = [row for row in v15["wheels"]
                      if row["normalized_name"] not in {"exceptiongroup", "tomli"}]
    old_locks = [text for name, text in zip(LOCK_NAMES, locks) if name != "BOOTSTRAP_REQUIREMENTS.txt"]
    old_locks[0] = "\n".join(line for line in old_locks[0].splitlines()
                              if not line.startswith(("exceptiongroup==", "tomli==")))
    result = validate_resolution(v15, old_locks)
    assert result["status"] == "failed"
    assert any("pytest requires exceptiongroup>=1" in item for item in result["errors"])
    assert any("pytest requires tomli>=1" in item for item in result["errors"])


def test_pytest_python310_conditional_closure_is_complete():
    evidence, locks = evidence_and_locks()
    assert validate_resolution(evidence, locks)["status"] == "passed"
    selected = {normalize(row["normalized_name"]): row for row in evidence["wheels"]
                if row.get("selected_for_lock")}
    pytest = selected["pytest"]
    environment = target_environment(evidence)
    active = {normalize(Requirement(raw).name) for raw in pytest["requires_dist"]
              if Requirement(raw).marker is None or Requirement(raw).marker.evaluate(environment)}
    assert {"exceptiongroup", "tomli"} <= active <= set(selected)
    assert "importlib-metadata" not in active
    assert selected["typing-extensions"]["version"] == "4.16.0"


def test_windows_only_colorama_is_not_in_linux_target_closure():
    evidence, locks = evidence_and_locks()
    assert "colorama" not in {normalize(row["normalized_name"]) for row in evidence["wheels"]
                              if row.get("selected_for_lock")}
    synthetic = copy.deepcopy(evidence)
    synthetic["status"] = "passed_static_resolution"
    synthetic["wheels"].append({
        "filename": "colorama-0.4.6-py2.py3-none-any.whl", "name": "colorama",
        "normalized_name": "colorama", "requires_dist": [], "requires_python": ">=2.7",
        "selected_for_lock": True,
        "sha256": "4f1d9991f5acc0ca119f9d443620b77f9d6b33703e51011c16baf57afb285fc6",
        "size_bytes": 25335, "version": "0.4.6",
    })
    result = validate_resolution(synthetic, locks + [
        "colorama==0.4.6 --hash=sha256:4f1d9991f5acc0ca119f9d443620b77f9d6b33703e51011c16baf57afb285fc6\n"])
    assert any("not in the target closure: ['colorama']" in item for item in result["errors"])


def test_production_lock_installs_are_no_deps_and_cannot_resolve_undeclared_packages():
    setup = (ROOT / "setup_environment.sh").read_text(encoding="utf-8")
    for variable in ("BOOTSTRAP_REQUIREMENTS", "TORCH_REQUIREMENT", "DGL_REQUIREMENT",
                     "RUNTIME_REQUIREMENTS", "PYTHON_LOCK"):
        line = next(item for item in setup.splitlines()
                    if ' -m pip install ' in item and f'"${variable}"' in item)
        assert "--no-deps" in line and "--require-hashes" in line
    assert '"$PY" -m pip check' in setup


def test_lock_stages_are_disjoint_and_bound_to_intended_sources():
    bootstrap = (ROOT / "BOOTSTRAP_REQUIREMENTS.txt").read_text(encoding="utf-8")
    ordinary = (ROOT / "PYTHON_DEPENDENCY_LOCK.txt").read_text(encoding="utf-8")
    torch = (ROOT / "TORCH_CU118_REQUIREMENT.txt").read_text(encoding="utf-8")
    dgl = (ROOT / "DGL_CU118_REQUIREMENT.txt").read_text(encoding="utf-8")
    runtime = (ROOT / "CUDA11_RUNTIME_REQUIREMENTS.txt").read_text(encoding="utf-8")
    assert "torch==" in torch and "torch==" not in ordinary
    assert "dgl @" in dgl and "dgl" not in ordinary.lower()
    assert all(name not in ordinary for name in (
        "nvidia-cuda-runtime-cu11", "nvidia-cublas-cu11",
        "nvidia-cusparse-cu11", "nvidia-cusolver-cu11"))
    assert all(name not in ordinary for name in ("pip==", "setuptools==", "wheel=="))
    assert all(name in runtime for name in (
        "nvidia-cuda-runtime-cu11", "nvidia-cublas-cu11",
        "nvidia-cusparse-cu11", "nvidia-cusolver-cu11"))
    assert all(name in bootstrap for name in ("pip==25.3", "setuptools==80.9.0", "wheel==0.45.1"))


def test_clean_install_evidence_is_exact_and_passed():
    if os.getenv("ALIGNN_BOOTSTRAP_TEST_MODE") == "1":
        placeholder = json.loads((ROOT / "CLEAN_INSTALL_EVIDENCE.json").read_text(encoding="utf-8"))
        assert placeholder["status"] == "pending_delftblue_exact_linux_validation"
        assert placeholder["release_blocked"] is True
        assert not os.getenv("ALIGNN_CLEAN_INSTALL_EVIDENCE")
        return
    evidence_path, transcript_path = resolve_paths()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["status"] == "passed_runtime_clean_install"
    assert evidence["target"] == {
        "python": "3.10.20", "pip": "25.3", "platform_system": "Linux",
        "platform_machine": "x86_64", "glibc": "2.28",
    }
    assert evidence["precertification_checks_exit_code"] == 0
    assert evidence["pip_check"] == "No broken requirements found."
    assert evidence["undeclared_distributions"] == [] and evidence["absent_distributions"] == []
    assert evidence["version_mismatches"] == {}
    assert all(evidence["imports"][name]["passed"] for name in ("torch", "dgl", "alignn"))
    assert all(evidence["sonames"][name]["loaded"] for name in (
        "libcudart.so.11.0", "libcublas.so.11", "libcusparse.so.11"))
    assert evidence["certification_ready"] is True
    assert validate_clean_install(evidence, transcript_path.read_bytes()) == []


def test_failure_marker_and_partial_prefix_contract_remains_fail_closed():
    setup = (ROOT / "setup_environment.sh").read_text(encoding="utf-8")
    assert 'rm -f "$MARKER"' in setup
    assert 'certification_marker_present":false' in setup
    assert 'printf \'{"environment_revision":9,"status":"failed"' in setup
    assert '[[ ! -x "$ENV_PREFIX/bin/python" || ! -f "$MARKER" ]]' in setup
    assert setup.index('create --prefix "$ENV_PREFIX" --clone "$STAGING_PREFIX"') < setup.rindex("certify_final_prefix")


def test_v5_v7_v8_and_unbound_v9_certifications_cannot_satisfy_v18(tmp_path):
    marker = tmp_path / "old.json"
    for revision in (5, 7, 8):
        marker.write_text(json.dumps({
            "status": "login_node_installation_certified",
            "a100_runtime_certified": False,
            "environment_revision": revision,
            "environment_prefix": str(tmp_path),
        }), encoding="utf-8")
        result = subprocess.run([
            sys.executable, str(ROOT / "scripts" / "require_certification.py"),
            "--kind", "login", "--path", str(marker), "--expected-prefix", str(tmp_path),
        ], text=True, capture_output=True, check=False)
        assert result.returncode != 0
    marker.write_text(json.dumps({
        "status": "login_node_installation_certified", "a100_runtime_certified": False,
        "environment_revision": 9, "environment_prefix": str(tmp_path),
    }), encoding="utf-8")
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "require_certification.py"),
        "--kind", "login", "--path", str(marker), "--expected-prefix", str(tmp_path)],
        text=True, capture_output=True, check=False)
    assert result.returncode != 0


def test_exact_pip_and_new_prefix_are_bound_everywhere():
    setup = (ROOT / "setup_environment.sh").read_text(encoding="utf-8")
    assert 'test "$("$PY" -m pip --version | awk \'{print $2}\')" = "25.3"' in setup
    assert "cu118_v9" in setup and "cu118_v8" not in setup
    for path in sorted((ROOT / "slurm").glob("*.sbatch")):
        source = path.read_text(encoding="utf-8")
        assert "cu118_v9" in source and "cu118_v8" not in source


def test_regeneration_tool_rejects_non_target_host_semantics():
    source = (ROOT / "scripts" / "record_dependency_resolution.py").read_text(encoding="utf-8")
    for token in ('platform.python_version() != "3.10.20"', 'platform.system() != "Linux"',
                  'platform.machine() != "x86_64"', 'version("pip") != "25.3"',
                  '"exceptiongroup", "tomli"'):
        assert token in source
