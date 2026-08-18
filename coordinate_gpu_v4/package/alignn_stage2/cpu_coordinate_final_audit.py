"""Persistent fail-closed audit for the 25-cell CPU coordinate experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration_contract import DEFAULT_APPROVAL, _load_approved_module
from .common import FOLDS, SEEDS, sha256_file, write_json
from .sigma_authorization import require_authorized_sigma


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--coordinate-cache-root", required=True)
    parser.add_argument("--oof-dir", required=True)
    args = parser.parse_args(argv)
    authorization = require_authorized_sigma()
    _, ts_hash = _load_approved_module(DEFAULT_APPROVAL)
    root, cache_root, oof_dir = map(Path, (args.work_root, args.coordinate_cache_root, args.oof_dir))
    failures, cells = [], []
    for fold in FOLDS:
        for seed in SEEDS:
            name = f"fold_{fold}/seed_{seed}"
            cell, cache = root / f"fold_{fold}" / f"seed_{seed}", cache_root / f"fold_{fold}" / f"seed_{seed}"
            try:
                complete = json.loads((cell / "COMPLETE.json").read_text(encoding="utf-8"))
                export = json.loads((cell / "CALIBRATION_EXPORT_STATUS.json").read_text(encoding="utf-8"))
                manifest = json.loads((cache / "COORDINATE_CACHE_MANIFEST.json").read_text(encoding="utf-8"))
                if complete.get("execution_device") != "cpu" or complete.get("coordinate_sigma_authorization_sha256") != authorization.authorization_sha256:
                    raise RuntimeError("completion authorization/device mismatch")
                if complete.get("approved_ts_sha256") != ts_hash or export.get("approved_ts_sha256") != ts_hash:
                    raise RuntimeError("temperature-scaler hash mismatch")
                if manifest.get("status") != "passed" or manifest.get("record_count") != 3000:
                    raise RuntimeError("coordinate cache not exhaustively certified")
                if manifest.get("authorization_sha256") != authorization.authorization_sha256:
                    raise RuntimeError("coordinate cache authorization mismatch")
                for branch, key in (("control", "control_checkpoint_sha256"),
                        ("random2_coordinate", "random2_coordinate_checkpoint_sha256")):
                    checkpoint = cell / branch / "best.pt"
                    if not checkpoint.is_file() or sha256_file(checkpoint) != complete.get(key):
                        raise RuntimeError(f"{branch} selected checkpoint mismatch")
                if (cell / "INCOMPLETE_RESUME_REQUIRED.json").exists():
                    raise RuntimeError("partial-resume marker remains")
                cells.append(name)
            except Exception as error:
                failures.append({"cell": name, "error": str(error)})
    oof_path = oof_dir / "CPU_COORDINATE_OOF_ANALYSIS.json"
    if not oof_path.is_file():
        failures.append({"cell": "OOF", "error": "CPU_COORDINATE_OOF_ANALYSIS.json missing"})
    else:
        oof = json.loads(oof_path.read_text(encoding="utf-8"))
        if oof.get("status") != "complete" or oof.get("smoke_and_profile_artifacts_included") is not False:
            failures.append({"cell": "OOF", "error": "OOF status/exclusion contract failed"})
    report = {"status": "passed" if not failures and len(cells) == 25 else "failed",
        "execution_device": "cpu", "valid_cells": len(cells), "expected_cells": 25,
        "trained_models": 2 * len(cells), "reported_readouts": 4 * len(cells),
        "coordinate_sigma_authorization_sha256": authorization.authorization_sha256,
        "approved_ts_sha256": ts_hash, "failures": failures,
        "outer_test_results_inspected_by_audit": False}
    write_json(oof_dir / "CPU_COORDINATE_FINAL_AUDIT.json", report)
    if report["status"] != "passed":
        raise RuntimeError(f"CPU coordinate final audit failed: {failures}")
    print("ALIGNN_CPU_COORDINATE_FINAL_AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
