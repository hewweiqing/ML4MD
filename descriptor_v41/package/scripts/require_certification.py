#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=("login", "a100"))
    parser.add_argument("--path", required=True)
    parser.add_argument("--expected-prefix")
    args = parser.parse_args()
    path = Path(args.path)
    if not path.is_file():
        raise SystemExit(f"required {args.kind} certification is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if args.kind == "login":
        passed = (value.get("status") == "login_node_installation_certified"
            and value.get("a100_runtime_certified") is False
            and value.get("environment_revision") == 9
            and value.get("certification_workflow_version") == 20
            and value.get("full_pytest_status") == "passed")
    else:
        passed = (value.get("status") == "passed" and value.get("a100_runtime_certified") is True
            and value.get("environment_revision") == 9)
    if args.expected_prefix:
        passed = passed and Path(value.get("environment_prefix", "")).resolve() == Path(args.expected_prefix).resolve()
    manifest = json.loads((ROOT / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    passed = passed and value.get("package_aggregate_sha256") == manifest.get("aggregate_sha256")
    if args.kind == "login" and passed:
        evidence = Path(value.get("clean_install_evidence_path", ""))
        transcript = Path(value.get("clean_install_transcript_path", ""))
        pytest_log = Path(value.get("full_pytest_log_path", ""))
        passed = all(item.is_file() for item in (evidence, transcript, pytest_log))
        passed = passed and sha256(evidence) == value.get("clean_install_evidence_sha256")
        passed = passed and sha256(transcript) == value.get("clean_install_transcript_sha256")
        passed = passed and sha256(pytest_log) == value.get("full_pytest_log_sha256")
    if args.kind == "a100" and passed and args.expected_prefix:
        login_marker = Path(args.expected_prefix) / ".alignn_stage2_v9_login_certification.json"
        passed = login_marker.is_file() and sha256(login_marker) == value.get("login_certification_sha256")
    if not passed:
        raise SystemExit(f"required {args.kind} certification is not passed: {path}")
    print(f"Required {args.kind} certification passed: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
