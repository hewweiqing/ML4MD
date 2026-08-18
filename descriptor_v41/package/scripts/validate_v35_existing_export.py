#!/usr/bin/env python3
"""Validate a single-use v34 export for v35 verification without rematerialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from alignn_stage2.calibration_contract import apply_temperature, fit_temperature
from alignn_stage2.common import sha256_file, write_json

V26_AGGREGATE = "1c15f877cee4eb6eebfd096962c6a793d0d3f14385287a6a9385e3adce05bd7d"
V26_MUBEN_SHA256 = "868654151b87d526f122f741183c23599913541d5d62aada5cb35c91bc510719"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v26-root", required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    v26_root, cell, output = Path(args.v26_root), Path(args.cell), Path(args.output)

    manifest = json.loads((v26_root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("aggregate_sha256") != V26_AGGREGATE:
        raise RuntimeError("immutable v26 package aggregate mismatch")
    if sha256_file(v26_root / "vendor/muben_temperature_scaling.py") != V26_MUBEN_SHA256:
        raise RuntimeError("immutable v26 MUBen source mismatch")
    sentinel = cell / "OUTER_TEST_ACCESS_STARTED.json"
    if not sentinel.is_file():
        raise RuntimeError("single-use outer-test sentinel is absent; existing-export recovery is invalid")

    required_export = [
        cell / "predictions/outer_test_predictions.csv",
        cell / "predictions/scaled_logits_sidecar.npz",
        cell / "temperature_scaling_provenance.json",
        cell / "split_provenance.json",
    ]
    missing = [str(path) for path in required_export if not path.is_file()]
    if missing:
        raise RuntimeError(f"single-use export is incomplete; refusing rematerialization: {missing}")

    fitted_rows, reference = {}, None
    for branch in ("control", "random2"):
        artifact_path = cell / branch / "validation_raw_logits.npz"
        if not artifact_path.is_file():
            raise RuntimeError(f"validation artifact is missing: {artifact_path}")
        with np.load(artifact_path, allow_pickle=False) as artifact:
            logits, labels = artifact["logits"], artifact["labels"]
            ids, order = artifact["structure_ids"], artifact["sample_order_index"]
            if logits.shape != (len(labels), 2):
                raise RuntimeError(f"{branch} validation logits are not native [n,2] logits")
            current = (ids.copy(), labels.copy(), order.copy())
            if reference is not None and not all(np.array_equal(a, b) for a, b in zip(reference, current)):
                raise RuntimeError("Control/Random2 validation alignment mismatch")
            reference = current
            fitted = fit_temperature(logits, labels, split="validation")
            scaled = apply_temperature(logits, fitted)
            if scaled.shape != logits.shape:
                raise RuntimeError("scaled-logit dimensions changed")
            fitted_rows[branch] = fitted.public_metadata()
            fitted_rows[branch]["validation_artifact_sha256"] = sha256_file(artifact_path)

    report = {
        "schema_version": 1,
        "status": "passed",
        "scope": "validation_recheck_and_existing_export_integrity_only",
        "outer_test_rematerialized": False,
        "outer_test_metrics_inspected_by_recovery_validator": False,
        "single_use_sentinel_present": True,
        "v26_package_aggregate_sha256": V26_AGGREGATE,
        "v26_muben_source_sha256": V26_MUBEN_SHA256,
        "cell": str(cell),
        "branches": fitted_rows,
        "existing_export_sha256": {str(path.relative_to(cell)): sha256_file(path) for path in required_export},
    }
    write_json(output, report)
    print(f"V35_EXISTING_EXPORT_RECOVERY_VALIDATION: PASS {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
