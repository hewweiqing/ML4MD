#!/usr/bin/env python3
"""Generate pre-certification clean-install evidence from the final environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def build_evidence(environment_report: dict, package_manifest: dict, resolution_sha256: str,
        transcript_path: Path) -> dict:
    pip_check = subprocess.run([sys.executable, "-m", "pip", "check"], text=True,
        capture_output=True, check=False)
    pip_text = pip_check.stdout.strip()
    libc_name, libc_version = platform.libc_ver()
    errors = list(environment_report.get("errors", []))
    if environment_report.get("status") != "login_node_installation_certified":
        errors.append("final-prefix environment report is not passed")
    if Path(environment_report.get("environment_prefix", "")).resolve() != Path(sys.prefix).resolve():
        errors.append("environment report prefix does not match the evidence interpreter")
    if pip_check.returncode != 0 or pip_text != "No broken requirements found.":
        errors.append(f"pip check failed: rc={pip_check.returncode}, stdout={pip_text}, stderr={pip_check.stderr.strip()}")
    target = {"python": platform.python_version(), "pip": version("pip"),
        "platform_system": platform.system(), "platform_machine": platform.machine(),
        "glibc": libc_version if libc_name == "glibc" else f"{libc_name}:{libc_version}"}
    expected_target = {"python": "3.10.20", "pip": "25.3", "platform_system": "Linux",
        "platform_machine": "x86_64", "glibc": "2.28"}
    if target != expected_target:
        errors.append(f"runtime target mismatch: expected {expected_target}, found {target}")
    sonames = {row["requested_soname"]: {"loaded": row["loader_succeeded"],
        "resolved_path": row["resolved_path"], "sha256": row["sha256"]}
        for row in environment_report.get("cuda_runtime", {}).get("libraries", [])}
    environment_report_sha256 = environment_report.get("report_file_sha256_pending")
    transcript = {"schema_version": 1, "purpose": "v20 final-prefix pre-certification transcript",
        "environment_prefix": str(Path(sys.prefix).resolve()), "target": target,
        "package_aggregate_sha256": package_manifest.get("aggregate_sha256"),
        "static_resolution_evidence_sha256": resolution_sha256,
        "environment_report_sha256": environment_report_sha256,
        "pip_check": {"returncode": pip_check.returncode, "stdout": pip_text,
            "stderr": pip_check.stderr.strip()},
        "checks": ["exact installed distribution closure", "required distribution file inventories",
            "CUDA 11 SONAME loading", "torch/DGL/ALIGNN imports", "pip check"]}
    transcript_bytes = (json.dumps(transcript, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_atomic(transcript_path, transcript_bytes)
    evidence = {"schema_version": 2,
        "status": "passed_runtime_clean_install" if not errors else "failed",
        "release_blocked": bool(errors), "target": target,
        "precertification_checks_exit_code": 0 if not errors else 1,
        "environment_prefix": str(Path(sys.prefix).resolve()), "pip_check": pip_text,
        "undeclared_distributions": environment_report.get("undeclared_distributions"),
        "absent_distributions": environment_report.get("absent_distributions"),
        "version_mismatches": environment_report.get("version_mismatches"),
        "imports": environment_report.get("imports", {}), "sonames": sonames,
        "certification_ready": not errors,
        "package_aggregate_sha256": package_manifest.get("aggregate_sha256"),
        "static_resolution_evidence_sha256": resolution_sha256,
        "environment_report_sha256": environment_report_sha256,
        "transcript_sha256": hashlib.sha256(transcript_bytes).hexdigest(), "errors": errors}
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-report", required=True)
    parser.add_argument("--package-manifest", required=True)
    parser.add_argument("--resolution-evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--transcript", required=True)
    args = parser.parse_args()
    report_path = Path(args.environment_report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["report_file_sha256_pending"] = sha256(report_path)
    manifest = json.loads(Path(args.package_manifest).read_text(encoding="utf-8"))
    evidence = build_evidence(report, manifest, sha256(Path(args.resolution_evidence)), Path(args.transcript))
    write_atomic(Path(args.output), (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["status"] == "passed_runtime_clean_install" else 1


if __name__ == "__main__":
    raise SystemExit(main())
