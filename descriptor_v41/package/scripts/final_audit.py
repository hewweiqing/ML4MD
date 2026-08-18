#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignn_stage2.common import FOLDS, SEEDS, write_json
from alignn_stage2.production import verify_complete


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--oof-dir", required=True)
    args = parser.parse_args()
    root, failures = Path(args.work_root), []
    cells = []
    for fold in FOLDS:
        for seed in SEEDS:
            cell = root / f"fold_{fold}" / f"seed_{seed}"
            if not verify_complete(cell, fold, seed):
                failures.append(f"fold_{fold}/seed_{seed}")
            else:
                cells.append(json.loads((cell / "COMPLETE.json").read_text(encoding="utf-8")))
    oof = Path(args.oof_dir) / "OOF_ANALYSIS.json"
    if not oof.is_file():
        failures.append("OOF_ANALYSIS.json")
    else:
        summary = json.loads(oof.read_text(encoding="utf-8"))
        if sorted(map(int, summary.get("per_seed", {}).keys())) != list(SEEDS):
            failures.append("five_complete_oof_seed_replicates")
    report = {"status": "passed" if not failures else "failed", "valid_cells": len(cells),
        "trained_branches": 2 * len(cells), "readouts": 4 * len(cells), "expected_cells": 25,
        "failures": failures}
    write_json(Path(args.oof_dir) / "FINAL_AUDIT.json", report)
    if failures or report["trained_branches"] != 50 or report["readouts"] != 100:
        raise RuntimeError(f"persistent final audit failed: {failures}")
    print("ALIGNN_PERSISTENT_FINAL_AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
