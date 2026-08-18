#!/usr/bin/env python3
"""Pre-run gate, called from the sbatch job (not from inside
init_diagnostic.py itself) before the full 20-seed Stage A run — matches
coordinate_gpu_v4's scripts/require_gpu_coordinate_gates.py pattern:
preflight must have passed, and if a profile is supplied it must be
accompanied by a matching human-reviewer approval bound to the current
package/profile identity hashes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from alignn_stage2 import stage_a_profile
from alignn_stage2.common import sha256_file


def _load(path):
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(f"required gate missing: {p}")
    return p, json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--approval")
    parser.add_argument("--package-manifest", default="PACKAGE_MANIFEST.json")
    parser.add_argument("--policy", default="STAGE_A_RESOURCE_POLICY.json")
    args = parser.parse_args()

    _, preflight = _load(args.preflight)
    if preflight.get("status") != "passed":
        raise RuntimeError("Stage A A100 preflight invalid")

    if args.profile or args.approval:
        if not (args.profile and args.approval):
            raise RuntimeError("profile and approval must be supplied together")
        profile_path, profile = _load(args.profile)
        _, approval = _load(args.approval)
        _, policy = _load(args.policy)
        current_bindings = stage_a_profile.binding_values(profile_path, Path(args.package_manifest))
        stage_a_profile.verify_approval(approval, profile, policy, current_bindings)

    print("REQUIRED_STAGE_A_GATES: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
