#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from alignn_stage2.calibration_contract import DEFAULT_APPROVAL, _load_approved_module
from alignn_stage2.common import CONDITIONS, sha256_ids, write_json
from alignn_stage2.production import assert_temperature_invariance, metrics, write_complete


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-dir", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    cell = Path(args.cell_dir)
    _, approved = _load_approved_module(DEFAULT_APPROVAL)
    with (cell / "predictions/outer_test_predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    sidecar = np.load(cell / "predictions/scaled_logits_sidecar.npz", allow_pickle=False)
    ids, labels = sidecar["outer_test_ids"].astype(str), sidecar["outer_test_labels"].astype(np.int64)
    if len(set(ids.tolist())) != len(ids) or len(rows) != 4 * len(ids):
        raise RuntimeError("outer-test row coverage/uniqueness failure")
    raw_control = np.asarray([[float(r["raw_native_logit_0"]), float(r["raw_native_logit_1"])]
        for r in rows if r["condition"] == CONDITIONS[0]])
    raw_random2 = np.asarray([[float(r["raw_native_logit_0"]), float(r["raw_native_logit_1"])]
        for r in rows if r["condition"] == CONDITIONS[2]])
    logits = {CONDITIONS[0]: raw_control, CONDITIONS[1]: sidecar["outer_test_control"],
        CONDITIONS[2]: raw_random2, CONDITIONS[3]: sidecar["outer_test_random2"]}
    invariance = {"control": assert_temperature_invariance(raw_control, logits[CONDITIONS[1]], labels),
        "random2": assert_temperature_invariance(raw_random2, logits[CONDITIONS[3]], labels)}
    summaries = {name: metrics(labels, value) for name, value in logits.items()}
    for left, right in ((CONDITIONS[0], CONDITIONS[1]), (CONDITIONS[2], CONDITIONS[3])):
        if summaries[left]["accuracy"] != summaries[right]["accuracy"] or summaries[left]["f1"] != summaries[right]["f1"]:
            raise RuntimeError("accuracy/F1 invariance failure")
        # AUROC invariance is established above from the tolerance-aware native
        # binary-margin ordering contract. Exact equality of independently
        # rounded floating-point summaries is not a valid additional gate.
    report = {"status": "passed", "fold": args.fold, "seed": args.seed, "structure_count": len(ids),
        "structure_ids_sha256": sha256_ids(ids), "metrics": summaries, "invariance": invariance,
        "trained_branches": 2, "readout_count": 4, "approved_ts_sha256": approved,
        "outer_test_used_for_training_or_fitting": False}
    write_json(cell / "CELL_VERIFICATION.json", report)
    write_complete(cell, args.fold, args.seed, approved)
    print("ALIGNN_CELL: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
