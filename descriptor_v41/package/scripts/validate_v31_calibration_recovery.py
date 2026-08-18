#!/usr/bin/env python3
"""Validate the v26 trained cell and revised scaler without outer-test access."""

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
    if (cell / "OUTER_TEST_ACCESS_STARTED.json").exists():
        raise RuntimeError("outer-test access sentinel exists; automatic recovery is prohibited")

    required = []
    for branch in ("control", "random2"):
        required.extend((cell / branch / "best.pt", cell / branch / "validation_raw_logits.npz",
            cell / branch / "checkpoint_provenance.json"))
    required.append(cell / "split_provenance.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"v26 trained-cell recovery inputs are missing: {missing}")

    fitted_rows, reference = {}, None
    for branch in ("control", "random2"):
        artifact_path = cell / branch / "validation_raw_logits.npz"
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
        "scope": "inner_validation_only",
        "outer_test_accessed": False,
        "v26_package_aggregate_sha256": V26_AGGREGATE,
        "v26_muben_source_sha256": V26_MUBEN_SHA256,
        "cell": str(cell),
        "branches": fitted_rows,
    }
    write_json(output, report)
    print(f"V31_CALIBRATION_RECOVERY_VALIDATION: PASS {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
