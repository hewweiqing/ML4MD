"""Fail-closed validation and human-approval binding for the Stage A
single-seed full-network-arm timing profile.

Not a copy of coordinate_gpu_v4's resource_profile.py: that schema encodes
assumptions specific to its own 25-cell paired-branch training grid
(measured_batches>=100, a persisted structure/coordinate shard cache,
control/random2 branch-hour projections, a 25-cell disk projection). Stage A
profiles a single thing — one seed of the full-network warm-up primitive,
per time_full_network_warmup_single_seed() in init_diagnostic.py — so this
module validates that report's actual shape instead of forcing it into an
unrelated schema. The same pattern (validate -> evaluate against a policy ->
require an explicit human-reviewer approval bound to current identity
hashes) is kept, since that pattern is what makes the gate meaningful, not
the specific field names.
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

from .common import sha256_file

PROFILE_SCHEMA_VERSION = 1
APPROVAL_SCHEMA_VERSION = 1


def validate_profile(report: dict) -> None:
    if report.get("status") != "passed":
        raise RuntimeError("Stage A single-seed profile is not a PASS report")
    if report.get("deterministic_replay") is not True:
        raise RuntimeError("Stage A single-seed profile lacks a verified deterministic replay")
    for name in ("elapsed_seconds", "elapsed_seconds_per_replay", "projected_seconds_for_20_seeds"):
        value = report.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"Stage A single-seed profile field must be finite and positive: {name}")
    if report.get("optimizer_steps_per_replay") != 938:
        raise RuntimeError("Stage A single-seed profile used a different optimizer-step budget than FULL_NETWORK_WARMUP_CONFIG.json")
    if report.get("batch_size") != 128:
        raise RuntimeError("Stage A single-seed profile used a different batch size than FULL_NETWORK_WARMUP_CONFIG.json")
    if not isinstance(report.get("package_aggregate_sha256"), str) or len(report["package_aggregate_sha256"]) != 64:
        raise RuntimeError("Stage A single-seed profile is missing its package_aggregate_sha256 identity digest")


def binding_values(profile_path: Path, package_manifest_path: Path) -> dict:
    import json
    package = json.loads(Path(package_manifest_path).read_text(encoding="utf-8"))
    return {"profile_sha256": sha256_file(profile_path), "package_aggregate_sha256": package["aggregate_sha256"]}


def evaluate_policy(profile: dict, policy: dict) -> dict:
    validate_profile(profile)
    max_hours = policy["projected_wall_time_hours_max"]
    projected_hours = profile["projected_seconds_for_20_seeds"] / 3600
    passed = math.isfinite(projected_hours) and 0 < projected_hours <= max_hours
    return {"passed": passed, "projected_wall_time_hours": projected_hours,
        "projected_wall_time_hours_max": max_hours}


def verify_approval(approval: dict, profile: dict, policy: dict, current_bindings: dict) -> None:
    validate_profile(profile)
    if approval.get("schema_version") != APPROVAL_SCHEMA_VERSION or approval.get("decision") != "approved":
        raise RuntimeError("Stage A profile has no explicit approved decision")
    if not approval.get("reviewer_identity") or not approval.get("reasons"):
        raise RuntimeError("Stage A profile approval lacks reviewer identity or reasons")
    try:
        datetime.fromisoformat(approval["approved_at_utc"].replace("Z", "+00:00"))
    except Exception as error:
        raise RuntimeError("invalid Stage A profile approval UTC time") from error
    if approval.get("bindings") != current_bindings:
        raise RuntimeError("Stage A profile approval is stale, copied, or mismatched")
    result = evaluate_policy(profile, policy)
    if not result["passed"] or approval.get("resource_policy_evaluation") != result:
        raise RuntimeError("Stage A profile approval policy evaluation failed or changed")
