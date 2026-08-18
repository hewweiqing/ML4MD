#!/usr/bin/env python3
"""Validate external gates for the v37 remaining-cell array without reading outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignn_stage2.production import verify_complete

V26_AGGREGATE = "1c15f877cee4eb6eebfd096962c6a793d0d3f14385287a6a9385e3adce05bd7d"
V36_AGGREGATE = "f28267cb8d6d4f094d33bee125c1fbf69dc908bfa637498f38129c8b8f513c38"


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"required external artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v26-root", required=True)
    parser.add_argument("--v36-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--final-root", required=True)
    args = parser.parse_args()
    v26, v36 = Path(args.v26_root), Path(args.v36_root)
    v26_manifest = read_json(v26 / "PACKAGE_MANIFEST.json")
    v36_manifest = read_json(v36 / "PACKAGE_MANIFEST.json")
    if v26_manifest.get("aggregate_sha256") != V26_AGGREGATE:
        raise RuntimeError("immutable v26 package aggregate mismatch")
    if v36_manifest.get("aggregate_sha256") != V36_AGGREGATE:
        raise RuntimeError("immutable v36 package aggregate mismatch")
    cache = read_json(Path(args.cache_root) / "STRUCTURE_CACHE_MANIFEST.json")
    if cache.get("status") != "passed" or cache.get("release_identity") != V26_AGGREGATE:
        raise RuntimeError("verified v26 structure-cache provenance mismatch")
    gate = read_json(v36 / "preflight/PRIMARY_FOLD0_GATE_PASSED.json")
    if gate.get("status") != "passed" or not gate.get("full_grid_authorized"):
        raise RuntimeError("v36 fold-0/seed-0 gate has not authorized the grid")
    cell = Path(args.final_root) / "fold_0" / "seed_0"
    if not verify_complete(cell, 0, 0):
        raise RuntimeError("promoted fold-0/seed-0 COMPLETE artifact is invalid")
    print("V37_EXTERNAL_FULL_GRID_INPUTS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
