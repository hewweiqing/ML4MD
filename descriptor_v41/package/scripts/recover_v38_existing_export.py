#!/usr/bin/env python3
"""Recover only the five v37 exports that failed after single-use export."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from alignn_stage2.calibration_contract import DEFAULT_APPROVAL, _load_approved_module
from alignn_stage2.common import CONDITIONS, sha256_file, write_json
from alignn_stage2.production import assert_temperature_invariance, write_complete

ALLOWED_CELLS = {(1, 4), (2, 0), (3, 0), (3, 2), (3, 4)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (args.fold, args.seed) not in ALLOWED_CELLS:
        raise RuntimeError("cell is not in the frozen v38 existing-export recovery set")
    cell = Path(args.cell)
    if (cell / "COMPLETE.json").exists():
        raise RuntimeError("cell is already complete; recovery refuses overwrite")
    if not (cell / "OUTER_TEST_ACCESS_STARTED.json").is_file():
        raise RuntimeError("single-use sentinel is required; ordinary pre-outer recovery must not use this path")
    status_path = cell / "CALIBRATION_EXPORT_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete" or status.get("outer_test_predictions_exported") is not True:
        raise RuntimeError("v37 calibration/export did not complete")
    _, approved_hash = _load_approved_module(DEFAULT_APPROVAL)
    if status.get("approved_ts_sha256") != approved_hash:
        raise RuntimeError("existing export used a different temperature-scaler hash")

    csv_path = cell / "predictions/outer_test_predictions.csv"
    sidecar_path = cell / "predictions/scaled_logits_sidecar.npz"
    required = [csv_path, sidecar_path, cell / "temperature_scaling_provenance.json",
        cell / "split_provenance.json", status_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"existing single-use export is incomplete: {missing}")
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    # Deliberately never interpret true_label or compute a metric in this recovery.
    with np.load(sidecar_path, allow_pickle=False) as sidecar:
        ids = sidecar["outer_test_ids"].astype(str)
        scaled_control = np.asarray(sidecar["outer_test_control"], dtype=np.float64)
        scaled_random2 = np.asarray(sidecar["outer_test_random2"], dtype=np.float64)
    if len(set(ids.tolist())) != len(ids) or len(rows) != 4 * len(ids):
        raise RuntimeError("existing export row coverage/identity failure")

    def raw(condition: str) -> np.ndarray:
        selected = [row for row in rows if row.get("condition") == condition]
        if len(selected) != len(ids):
            raise RuntimeError(f"{condition}: row count mismatch")
        return np.asarray([[float(row["raw_native_logit_0"]), float(row["raw_native_logit_1"])]
            for row in selected], dtype=np.float64)

    invariance = {
        "control": assert_temperature_invariance(raw(CONDITIONS[0]), scaled_control),
        "random2": assert_temperature_invariance(raw(CONDITIONS[2]), scaled_random2),
    }
    report = {"schema_version": 1, "status": "passed", "fold": args.fold, "seed": args.seed,
        "scope": "existing_single_use_export_structure_hash_and_margin_invariance_only",
        "outer_test_rematerialized": False, "outer_test_labels_interpreted": False,
        "outer_test_metrics_computed_or_compared": False, "approved_ts_sha256": approved_hash,
        "structure_count": len(ids), "invariance": invariance,
        "artifact_sha256": {str(path.relative_to(cell)): sha256_file(path) for path in required}}
    write_json(Path(args.output), report)
    write_json(cell / "CELL_VERIFICATION.json", report)
    write_complete(cell, args.fold, args.seed, approved_hash)
    print("V38_EXISTING_EXPORT_RECOVERY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
