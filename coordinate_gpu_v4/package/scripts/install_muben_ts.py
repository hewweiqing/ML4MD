#!/usr/bin/env python3
"""Install one validated external MUBen TS source; does not implement TS."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np

from alignn_stage2.common import sha256_file
from alignn_stage2.calibration_contract import apply_temperature, fit_temperature


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--implementation-version", required=True)
    parser.add_argument("--final-audit", required=True)
    parser.add_argument("--final-audit-sha256", required=True)
    args = parser.parse_args()
    source = Path(args.source).resolve()
    if not source.is_file() or len(args.expected_sha256) != 64:
        raise SystemExit("source file and 64-character expected SHA-256 are required")
    actual = sha256_file(source)
    if actual != args.expected_sha256:
        raise SystemExit(f"refusing install: expected {args.expected_sha256}, found {actual}")
    final_audit = Path(args.final_audit).resolve()
    if not final_audit.is_file() or sha256_file(final_audit) != args.final_audit_sha256:
        raise SystemExit("final MUBen audit is absent or hash-mismatched")
    audit_value = json.loads(final_audit.read_text(encoding="utf-8"))
    if audit_value.get("result", audit_value.get("status")) != "PASS":
        raise SystemExit("final MUBen audit did not PASS")
    spec = importlib.util.spec_from_file_location("muben_ts_install_check", source)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot import source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "fit_temperature", None)) or not callable(getattr(module, "apply_temperature", None)):
        raise SystemExit("source does not export fit_temperature and apply_temperature")
    root = Path(__file__).resolve().parents[1]
    destination = root / "vendor" / "muben_temperature_scaling.py"
    temporary = destination.with_suffix(".py.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)
    approval = {"schema_version": 3, "status": "approved_muben_persistent_final_audit_passed", "installed_source": "vendor/muben_temperature_scaling.py",
        "expected_sha256": args.expected_sha256, "implementation_version": args.implementation_version,
        "source_origin": str(source), "muben_final_audit_sha256": args.final_audit_sha256,
        "policy": "exact source digest and final PASS audit required before calibration or outer-test access"}
    (root / "MUBEN_TS_APPROVAL.json").write_text(json.dumps(approval, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if sha256_file(destination) != args.expected_sha256:
        raise SystemExit("post-install digest verification failed")
    synthetic_logits = np.asarray([[-2.0, 1.0], [3.0, -1.0], [0.25, 0.1], [-0.4, 0.8]], dtype=np.float64)
    synthetic_labels = np.asarray([1, 0, 0, 1], dtype=np.int64)
    try:
        fitted_1 = fit_temperature(synthetic_logits, synthetic_labels)
        fitted_2 = fit_temperature(synthetic_logits.copy(), synthetic_labels.copy())
        scaled_1 = apply_temperature(synthetic_logits, fitted_1)
        scaled_2 = apply_temperature(synthetic_logits, fitted_2)
        if fitted_1.public_metadata() != fitted_2.public_metadata() or not np.array_equal(scaled_1, scaled_2):
            raise RuntimeError("deterministic synthetic repeatability failed")
    except Exception as error:
        blocked = {"schema_version": 1, "status": "blocked_failed_synthetic_contract_verification",
            "installed_source": "vendor/muben_temperature_scaling.py", "expected_sha256": None,
            "implementation_version": args.implementation_version, "error": f"{type(error).__name__}: {error}"}
        (root / "MUBEN_TS_APPROVAL.json").write_text(json.dumps(blocked, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    print(f"Installed validated MUBen TS {args.implementation_version}: {args.expected_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
