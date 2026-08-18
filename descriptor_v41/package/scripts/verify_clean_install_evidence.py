#!/usr/bin/env python3
"""Validate runtime clean-install evidence stored outside the immutable package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TARGET = {
    "python": "3.10.20", "pip": "25.3", "platform_system": "Linux",
    "platform_machine": "x86_64", "glibc": "2.28",
}
REQUIRED_IMPORTS = ("torch", "dgl", "alignn")
REQUIRED_SONAMES = ("libcudart.so.11.0", "libcublas.so.11", "libcusparse.so.11")


def validate(evidence: dict, transcript: bytes) -> list[str]:
    errors: list[str] = []
    if evidence.get("status") != "passed_runtime_clean_install" or evidence.get("release_blocked") is not False:
        errors.append("runtime clean installation is not passed")
    if evidence.get("target") != EXPECTED_TARGET:
        errors.append("clean-install target mismatch")
    if evidence.get("precertification_checks_exit_code") != 0:
        errors.append("precertification checks did not exit zero")
    if evidence.get("pip_check") != "No broken requirements found.":
        errors.append("pip check did not pass")
    if evidence.get("undeclared_distributions") != [] or evidence.get("absent_distributions") != []:
        errors.append("installed distributions do not exactly match the declared closure")
    if evidence.get("version_mismatches") != {}:
        errors.append("installed distribution versions differ from the declared closure")
    if evidence.get("transcript_sha256") != hashlib.sha256(transcript).hexdigest():
        errors.append("clean-install transcript hash mismatch")
    for name in REQUIRED_IMPORTS:
        if evidence.get("imports", {}).get(name, {}).get("passed") is not True:
            errors.append(f"required import not passed: {name}")
    for name in REQUIRED_SONAMES:
        if evidence.get("sonames", {}).get(name, {}).get("loaded") is not True:
            errors.append(f"required SONAME not loaded: {name}")
    if evidence.get("certification_ready") is not True:
        errors.append("runtime evidence is not ready for final certification")
    if len(evidence.get("package_aggregate_sha256", "")) != 64:
        errors.append("package aggregate is not bound into clean-install evidence")
    if len(evidence.get("static_resolution_evidence_sha256", "")) != 64:
        errors.append("static resolution evidence is not bound into clean-install evidence")
    if len(evidence.get("environment_report_sha256", "")) != 64:
        errors.append("environment report is not bound into clean-install evidence")
    return errors


def resolve_paths(evidence: str | None = None, transcript: str | None = None) -> tuple[Path, Path]:
    evidence_value = evidence or os.getenv("ALIGNN_CLEAN_INSTALL_EVIDENCE")
    transcript_value = transcript or os.getenv("ALIGNN_CLEAN_INSTALL_TRANSCRIPT")
    if not evidence_value or not transcript_value:
        raise RuntimeError("external runtime evidence paths must be supplied explicitly or through ALIGNN_CLEAN_INSTALL_EVIDENCE/TRANSCRIPT")
    return Path(evidence_value), Path(transcript_value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence")
    parser.add_argument("--transcript")
    args = parser.parse_args()
    try:
        evidence_path, transcript_path = resolve_paths(args.evidence, args.transcript)
        value = json.loads(evidence_path.read_text(encoding="utf-8"))
        transcript = transcript_path.read_bytes()
        errors = validate(value, transcript)
    except Exception as error:
        errors = [f"runtime evidence unavailable: {type(error).__name__}: {error}"]
    print(json.dumps({"status": "passed" if not errors else "failed", "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
