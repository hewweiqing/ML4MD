#!/usr/bin/env python3
"""Validation-only diagnostics for the three v37 nonconvergent cells."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from alignn_stage2.calibration_contract import DEFAULT_APPROVAL, _load_approved_module
from alignn_stage2.common import sha256_file, write_json

ALLOWED = {(1, 1): "control", (2, 4): "random2", (3, 1): "control"}
PUBLIC = ("temperature", "objective", "converged", "convergence_status", "optimization_steps",
    "validation_nll_before", "validation_nll_after", "final_abs_log_T_gradient", "implementation_version",
    "parameterization", "numerical_dtype", "optimizer", "max_iterations", "tolerance_grad",
    "tolerance_change", "fit_split")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    branch = ALLOWED.get((args.fold, args.seed))
    if branch is None:
        raise RuntimeError("cell is not in the frozen v38 pre-outer diagnostic set")
    cell = Path(args.cell)
    if (cell / "OUTER_TEST_ACCESS_STARTED.json").exists():
        raise RuntimeError("outer-test sentinel exists; validation-only diagnostic refuses this cell")
    module, digest = _load_approved_module(DEFAULT_APPROVAL)
    results = {}
    for name in ("control", "random2"):
        path = cell / name / "validation_raw_logits.npz"
        with np.load(path, allow_pickle=False) as artifact:
            fitted = module.fit_temperature(artifact["logits"], artifact["labels"])
        results[name] = {key: fitted.get(key) for key in PUBLIC}
        results[name]["validation_artifact_sha256"] = sha256_file(path)
    report = {"schema_version": 1, "status": "blocked_pending_validated_muben_numerical_resolution",
        "fold": args.fold, "seed": args.seed, "original_failing_branch": branch,
        "scope": "inner_validation_logits_only", "outer_test_accessed": False,
        "approved_module_sha256": digest, "fits": results,
        "policy": "No tolerance relaxation, optimizer substitution, or alternative fitter is authorized by v38."}
    write_json(Path(args.output), report)
    print("V38_PREOUTER_CONVERGENCE_DIAGNOSTIC: RECORDED_AND_BLOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
