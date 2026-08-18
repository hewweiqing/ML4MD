#!/usr/bin/env python3
"""Require an exact, current profile plus explicit resource approval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignn_stage2.resource_profile import binding_values, verify_approval


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--certification", required=True)
    parser.add_argument("--package-manifest", required=True)
    parser.add_argument("--execution-plan", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--resource-policy", required=True)
    parser.add_argument("--primary-slurm", required=True)
    args = parser.parse_args()
    names = ("profile", "approval", "certification", "package_manifest", "execution_plan",
        "cache_manifest", "resource_policy", "primary_slurm")
    paths = {name: Path(getattr(args, name)) for name in names}
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"required profile/approval evidence is missing: {name}={path}")
    profile = json.loads(paths["profile"].read_text(encoding="utf-8"))
    approval = json.loads(paths["approval"].read_text(encoding="utf-8"))
    policy = json.loads(paths["resource_policy"].read_text(encoding="utf-8"))
    certification = json.loads(paths["certification"].read_text(encoding="utf-8"))
    cache_manifest = json.loads(paths["cache_manifest"].read_text(encoding="utf-8"))
    if certification.get("status") != "passed" or certification.get("a100_runtime_certified") is not True:
        raise RuntimeError("bound A100 runtime certification is not PASS")
    if cache_manifest.get("status") != "passed" or cache_manifest.get("exhaustive_verification", {}).get("status") != "passed":
        raise RuntimeError("bound structure-cache manifest is not exhaustively verified PASS evidence")
    bindings = binding_values(paths["profile"], paths["certification"], paths["package_manifest"],
        paths["execution_plan"], paths["cache_manifest"], paths["resource_policy"], paths["primary_slurm"])
    verify_approval(approval, profile, policy, bindings)
    print("ALIGNN_PROFILE_APPROVAL: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
