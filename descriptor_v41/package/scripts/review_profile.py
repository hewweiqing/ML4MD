#!/usr/bin/env python3
"""Explicit operator review; never bundled with a pre-approved decision."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from alignn_stage2.common import write_json
from alignn_stage2.resource_profile import (APPROVAL_SCHEMA_VERSION, binding_values,
    evaluate_policy, validate_profile)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--certification", required=True)
    parser.add_argument("--package-manifest", required=True)
    parser.add_argument("--execution-plan", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--resource-policy", required=True)
    parser.add_argument("--primary-slurm", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--decision", choices=("approve", "reject"), required=True)
    parser.add_argument("--reason", action="append", required=True)
    args = parser.parse_args()
    paths = {name: Path(getattr(args, name.replace("-", "_"))) for name in
        ("profile", "certification", "package_manifest", "execution_plan", "cache_manifest", "resource_policy", "primary_slurm")}
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"review input is missing: {name}={path}")
    profile = json.loads(paths["profile"].read_text(encoding="utf-8"))
    certification = json.loads(paths["certification"].read_text(encoding="utf-8"))
    policy = json.loads(paths["resource_policy"].read_text(encoding="utf-8"))
    validate_profile(profile)
    if certification.get("status") != "passed" or certification.get("a100_runtime_certified") is not True:
        raise RuntimeError("A100 runtime certification has not passed")
    bindings = binding_values(paths["profile"], paths["certification"], paths["package_manifest"],
        paths["execution_plan"], paths["cache_manifest"], paths["resource_policy"], paths["primary_slurm"])
    for key, value in bindings.items():
        if key != "profile_sha256" and profile.get(key) != value:
            raise RuntimeError(f"profile is not bound to current evidence: {key}")
    if profile.get("requested_slurm_resources") != policy.get("requested_slurm_resources"):
        raise RuntimeError("profile requested-resource record differs from policy")
    evaluation = evaluate_policy(profile, policy)
    decision = "approved" if args.decision == "approve" else "rejected"
    if decision == "approved" and not evaluation["passed"]:
        raise RuntimeError("resource policy failed; approval is prohibited")
    record = {"schema_version": APPROVAL_SCHEMA_VERSION, "decision": decision,
        "reviewer_identity": args.reviewer, "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "reasons": args.reason, "bindings": bindings, "resource_policy_evaluation": evaluation,
        "generated_on_delftblue_runtime": True, "slurm_job_id": os.getenv("SLURM_JOB_ID")}
    output = Path(args.output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    write_json(temporary, record)
    os.replace(temporary, output)
    print(f"PROFILE REVIEW: {decision.upper()}")
    return 0 if decision == "approved" else 2


if __name__ == "__main__":
    raise SystemExit(main())
