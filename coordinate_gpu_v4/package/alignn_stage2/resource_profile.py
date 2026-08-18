"""Fail-closed validation and approval bindings for the v11 resource profile."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .common import sha256_file

PROFILE_SCHEMA_VERSION = 2
APPROVAL_SCHEMA_VERSION = 1


def _get(value: dict, dotted: str):
    current = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise RuntimeError(f"resource profile missing required field: {dotted}")
        current = current[part]
    return current


def _positive(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"resource profile field must be finite and positive: {name}")
    return float(value)


def validate_profile(report: dict) -> None:
    if report.get("schema_version") != PROFILE_SCHEMA_VERSION or report.get("status") != "passed":
        raise RuntimeError("resource profile is not a schema-v2 PASS report")
    for name in ("package_aggregate_sha256", "dataset_sha256", "split_identity_sha256",
            "cache_manifest_sha256", "cache_provenance_sha256", "a100_runtime_certification_sha256",
            "execution_plan_sha256", "resource_policy_sha256", "primary_slurm_sha256"):
        value = report.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"invalid profile identity digest: {name}")
    required_positive = (
        "cache.total_verified_load_seconds", "cache.manifest_lookup_seconds",
        "cache.hash_verification_seconds", "cache.graph_loading_seconds",
        "cache.requested_structures", "cache.requested_shards",
        "initialization.model_optimizer_scheduler_seconds", "training.measured_batches",
        "training.total_measured_seconds", "training.seconds_per_batch",
        "training.forward_seconds", "training.backward_seconds", "training.optimizer_seconds",
        "training.scheduler_seconds", "random2.descriptor_measured_seconds",
        "random2.descriptor_projected_seconds", "random2.warmup_replay_measured_seconds",
        "validation.representative_full_validation_seconds", "validation.scheduled_evaluations_all_branches",
        "io.checkpoint_history_measured_seconds", "memory.peak_cuda_allocated_bytes",
        "memory.peak_cuda_reserved_bytes", "memory.detected_gpu_total_bytes",
        "memory.process_peak_rss_bytes", "memory.projected_host_peak_bytes",
        "disk.cache_size_bytes", "disk.projected_cell_output_bytes", "disk.projected_full_grid_bytes",
        "disk.available_bytes", "projections.control_branch_hours", "projections.random2_branch_hours",
        "projections.paired_cell_hours", "projections.conservative_paired_upper_bound_hours",
        "projections.time_safety_factor")
    for name in required_positive:
        _positive(_get(report, name), name)
    if _get(report, "training.measured_batches") < 100:
        raise RuntimeError("resource profile has fewer than 100 measured optimizer batches")
    if report.get("training", {}).get("finite_losses") is not True:
        raise RuntimeError("resource profile contains non-finite training losses")
    if report.get("random2", {}).get("deterministic_replay_verified") is not True:
        raise RuntimeError("resource profile lacks deterministic Random2 replay verification")
    if _get(report, "cache.verified_shards") != _get(report, "cache.requested_shards"):
        raise RuntimeError("cache profile contains unverified requested shards")
    if _get(report, "cache.memory_cache_hits") + _get(report, "cache.memory_cache_misses") != _get(report, "cache.requested_shards"):
        raise RuntimeError("cache hit/miss accounting is contradictory")
    if _get(report, "memory.peak_cuda_reserved_bytes") < _get(report, "memory.peak_cuda_allocated_bytes"):
        raise RuntimeError("CUDA reserved memory is smaller than allocated memory")
    if _get(report, "memory.detected_gpu_total_bytes") < _get(report, "memory.peak_cuda_reserved_bytes"):
        raise RuntimeError("peak CUDA reservation exceeds detected device memory")
    if _get(report, "disk.projected_full_grid_bytes") < 25 * _get(report, "disk.projected_cell_output_bytes"):
        raise RuntimeError("full-grid disk projection omits one or more cells")
    control = _get(report, "projections.control_branch_hours")
    random2 = _get(report, "projections.random2_branch_hours")
    paired = _get(report, "projections.paired_cell_hours")
    cache_hours = _get(report, "cache.total_verified_load_seconds") / 3600
    if paired + 1e-12 < control + random2 + cache_hours:
        raise RuntimeError("paired runtime does not include both branches plus cache overhead")
    expected_upper = paired * _get(report, "projections.time_safety_factor")
    if not math.isclose(_get(report, "projections.conservative_paired_upper_bound_hours"), expected_upper,
            rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError("conservative paired upper-bound formula is contradictory")
    measured = set(report.get("measured_phases", []))
    if "graph_cache_verified_load" not in measured or "100_optimizer_batches" not in measured:
        raise RuntimeError("profile omits required measured-phase declarations")
    if "structure_graph_construction" in measured:
        raise RuntimeError("profile falsely claims graph construction was measured")
    if not isinstance(report.get("excluded_phases"), list) or not report["excluded_phases"]:
        raise RuntimeError("profile must explicitly declare excluded phases")
    if not isinstance(report.get("calculation_formulas"), dict) or not report["calculation_formulas"]:
        raise RuntimeError("profile calculation formulas are missing")


def binding_values(profile_path: Path, certification_path: Path, package_manifest_path: Path,
        execution_plan_path: Path, cache_manifest_path: Path, resource_policy_path: Path,
        primary_slurm_path: Path) -> dict:
    package = json.loads(Path(package_manifest_path).read_text(encoding="utf-8"))
    return {"profile_sha256": sha256_file(profile_path),
        "a100_runtime_certification_sha256": sha256_file(certification_path),
        "package_aggregate_sha256": package["aggregate_sha256"],
        "execution_plan_sha256": sha256_file(execution_plan_path),
        "cache_manifest_sha256": sha256_file(cache_manifest_path),
        "resource_policy_sha256": sha256_file(resource_policy_path),
        "primary_slurm_sha256": sha256_file(primary_slurm_path)}


def evaluate_policy(profile: dict, policy: dict) -> dict:
    validate_profile(profile)
    thresholds, requested = policy["approval_thresholds"], policy["requested_slurm_resources"]
    ratios = {
        "runtime_fraction": profile["projections"]["conservative_paired_upper_bound_hours"] / requested["primary_wall_time_hours"],
        "host_memory_fraction": profile["memory"]["projected_host_peak_bytes"] / requested["host_memory_bytes"],
        "gpu_reserved_fraction": profile["memory"]["peak_cuda_reserved_bytes"] / profile["memory"]["detected_gpu_total_bytes"],
        "disk_fraction": profile["disk"]["projected_full_grid_bytes"] / profile["disk"]["available_bytes"],
    }
    limits = {"runtime_fraction": thresholds["runtime_fraction_max"],
        "host_memory_fraction": thresholds["host_memory_fraction_max"],
        "gpu_reserved_fraction": thresholds["gpu_reserved_fraction_max"],
        "disk_fraction": thresholds["disk_fraction_max"]}
    checks = {name: math.isfinite(value) and value > 0 and value <= limits[name] for name, value in ratios.items()}
    return {"passed": all(checks.values()), "ratios": ratios, "limits": limits, "checks": checks}


def verify_approval(approval: dict, profile: dict, policy: dict, current_bindings: dict) -> None:
    validate_profile(profile)
    if approval.get("schema_version") != APPROVAL_SCHEMA_VERSION or approval.get("decision") != "approved":
        raise RuntimeError("runtime profile has no explicit approved decision")
    if not approval.get("reviewer_identity") or not approval.get("reasons"):
        raise RuntimeError("profile approval lacks reviewer identity or reasons")
    try:
        datetime.fromisoformat(approval["approved_at_utc"].replace("Z", "+00:00"))
    except Exception as error:
        raise RuntimeError("invalid profile approval UTC time") from error
    if approval.get("bindings") != current_bindings:
        raise RuntimeError("profile approval is stale, copied, or mismatched")
    profile_bindings = {key: profile[key] for key in current_bindings if key != "profile_sha256"}
    expected_profile_bindings = {key: value for key, value in current_bindings.items() if key != "profile_sha256"}
    if profile_bindings != expected_profile_bindings:
        raise RuntimeError("profile identities do not match current certification/package/cache/resources")
    result = evaluate_policy(profile, policy)
    if not result["passed"] or approval.get("resource_policy_evaluation") != result:
        raise RuntimeError("profile approval policy evaluation failed or changed")
