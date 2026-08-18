#!/usr/bin/env python3
"""Create a validation-only deterministic resolution record for one exact v40 cell."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from alignn_stage2.calibration_contract import DEFAULT_APPROVAL, _load_approved_module
from alignn_stage2.common import sha256_file, write_json
from alignn_stage2.numerical_resolution import assess_fit, cell_key, load_authorization


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    key = cell_key(args.fold, args.seed)
    cell = Path(args.cell)
    if (cell / "OUTER_TEST_ACCESS_STARTED.json").exists():
        raise RuntimeError("outer-test sentinel exists; v40 numerical resolution refuses this cell")
    authorization, authorization_sha256 = load_authorization()
    module, digest = _load_approved_module(DEFAULT_APPROVAL)
    if digest != authorization["muben_source_sha256"]:
        raise RuntimeError("approved MUBen source does not match v40 authorization")
    expected = authorization["authorized_cells"][key]
    fits = {}
    for branch in ("control", "random2"):
        artifact_path = cell / branch / "validation_raw_logits.npz"
        artifact_sha256 = sha256_file(artifact_path)
        if artifact_sha256 != expected[f"{branch}_validation_artifact_sha256"]:
            raise RuntimeError(f"{branch} validation artifact is not the authorized immutable input")
        with np.load(artifact_path, allow_pickle=False) as artifact:
            first = module.fit_temperature(artifact["logits"], artifact["labels"])
            second = module.fit_temperature(artifact["logits"], artifact["labels"])
        assessed = assess_fit(first, second, authorization["muben_source_sha256"], digest)
        assessed["validation_artifact_sha256"] = artifact_sha256
        if not assessed["accepted"]:
            raise RuntimeError(f"{branch} failed the authorized v40 numerical-resolution criteria: {assessed['checks']}")
        fits[branch] = assessed
    report = {
        "schema_version": 1,
        "status": "passed_v40_numerical_resolution",
        "fold": args.fold,
        "seed": args.seed,
        "scope": "post_training_pre_outer_test_inner_validation_only",
        "outer_test_accessed": False,
        "authorization_sha256": authorization_sha256,
        "approved_module_sha256": digest,
        "original_convergence_results_preserved": True,
        "fits": fits,
    }
    write_json(Path(args.output), report)
    print(f"V40_NUMERICAL_RESOLUTION: PASS {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
