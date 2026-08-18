#!/usr/bin/env python3
"""Authorize only the observed v19 Conda-clone bootstrap-tool overlay for recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_ERRORS = {
    "setuptools: expected 80.9.0, found 84.0.0",
    "wheel: expected 0.45.1, found 0.47.0",
    "installed distribution versions differ from closure: {'pip': {'expected': '25.3', 'actual': '26.2.1'}}",
}


def validate(value: dict, expected_prefix: Path) -> list[str]:
    errors: list[str] = []
    if value.get("status") != "failed" or value.get("environment_revision") != 9:
        errors.append("report is not a failed revision-9 final-prefix report")
    for key in ("environment_prefix", "expected_prefix"):
        if Path(value.get(key, "")).resolve() != expected_prefix.resolve():
            errors.append(f"{key} does not match the expected final prefix")
    if set(value.get("errors", [])) != EXPECTED_ERRORS:
        errors.append("failure set is not the exact observed v19 Conda-clone overlay")
    if value.get("undeclared_distributions") != [] or value.get("absent_distributions") != []:
        errors.append("distribution closure has missing or undeclared packages")
    if value.get("version_mismatches") != {
            "pip": {"expected": "25.3", "actual": "26.2.1"}}:
        errors.append("pip inventory mismatch is not the observed clone overlay")
    versions = value.get("versions", {})
    if versions.get("setuptools") != "84.0.0" or versions.get("wheel") != "0.47.0":
        errors.append("setuptools/wheel versions are not the observed clone overlay")
    for name in ("torch", "dgl", "alignn"):
        if value.get("imports", {}).get(name, {}).get("passed") is not True:
            errors.append(f"required import failed before recovery: {name}")
    if value.get("cuda_runtime", {}).get("passed") is not True:
        errors.append("CUDA runtime was not passed before recovery")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-prefix", required=True)
    args = parser.parse_args()
    value = json.loads(Path(args.report).read_text(encoding="utf-8"))
    errors = validate(value, Path(args.expected_prefix))
    print(json.dumps({"status": "passed" if not errors else "failed", "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
