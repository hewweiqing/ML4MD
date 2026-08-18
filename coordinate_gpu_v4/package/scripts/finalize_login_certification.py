#!/usr/bin/env python3
"""Run final pytest and atomically issue the hash-bound login certification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from alignn_stage2.common import verify_manifest
from scripts.verify_clean_install_evidence import validate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-report", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--tests", required=True)
    parser.add_argument("--pytest-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    package_root = Path(args.package_root).resolve()
    evidence_path, transcript_path = Path(args.evidence).resolve(), Path(args.transcript).resolve()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    errors = validate(evidence, transcript_path.read_bytes())
    package_verification = verify_manifest(package_root)
    if package_verification["status"] != "passed":
        errors.append("immutable package verification failed before final pytest")
    if evidence.get("package_aggregate_sha256") != package_verification.get("actual_aggregate"):
        errors.append("runtime evidence package aggregate does not match the immutable package")
    if errors:
        raise SystemExit("; ".join(errors))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("ALIGNN_BOOTSTRAP_TEST_MODE", None)
    env["ALIGNN_CLEAN_INSTALL_EVIDENCE"] = str(evidence_path)
    env["ALIGNN_CLEAN_INSTALL_TRANSCRIPT"] = str(transcript_path)
    command = [sys.executable, "-m", "pytest", "--assert=plain", "-q", "-p", "no:cacheprovider", args.tests]
    completed = subprocess.run(command, cwd=package_root, env=env, text=True, capture_output=True, check=False)
    pytest_log = Path(args.pytest_log).resolve()
    pytest_log.parent.mkdir(parents=True, exist_ok=True)
    pytest_log.write_text(json.dumps({"command": command, "returncode": completed.returncode,
        "stdout": completed.stdout, "stderr": completed.stderr}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if completed.returncode != 0:
        raise SystemExit(f"final runtime pytest failed; inspect {pytest_log}")
    report_path = Path(args.environment_report).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "login_node_installation_certified" or report.get("errors"):
        raise SystemExit("provisional final-prefix environment report is not passed")
    report.update({"certification_workflow_version": 20,
        "package_aggregate_sha256": package_verification["actual_aggregate"],
        "clean_install_evidence_path": str(evidence_path),
        "clean_install_evidence_sha256": sha256(evidence_path),
        "clean_install_transcript_path": str(transcript_path),
        "clean_install_transcript_sha256": sha256(transcript_path),
        "full_pytest_status": "passed", "full_pytest_log_path": str(pytest_log),
        "full_pytest_log_sha256": sha256(pytest_log)})
    write_atomic(Path(args.output).resolve(), report)
    print(json.dumps({"status": "passed", "certification": str(Path(args.output).resolve()),
        "package_aggregate_sha256": package_verification["actual_aggregate"],
        "pytest_log_sha256": sha256(pytest_log)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
