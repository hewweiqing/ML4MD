"""Fail-closed v40 numerical-resolution policy for exact pre-outer cells."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .common import sha256_file

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTHORIZATION = PACKAGE_ROOT / "V40_NUMERICAL_RESOLUTION_AUTHORIZATION.json"
FINAL_STATUS = "authorized_v40_post_training_pre_outer_test_numerical_resolution"
RESOLUTION_THRESHOLD = 1e-6
AUTHORIZED_CELLS = {(1, 1), (2, 4), (3, 1)}
FIT_FIELDS = (
    "temperature", "objective", "converged", "convergence_status",
    "optimization_steps", "validation_nll_before", "validation_nll_after",
    "final_abs_log_T_gradient", "implementation_version", "parameterization",
    "numerical_dtype", "optimizer", "max_iterations", "tolerance_grad",
    "tolerance_change", "fit_split",
)


class NumericalResolutionError(RuntimeError):
    """Raised when the v40 amendment cannot be applied exactly."""


def load_authorization(path: str | Path = DEFAULT_AUTHORIZATION) -> tuple[dict[str, Any], str]:
    path = Path(path)
    if not path.is_file():
        raise NumericalResolutionError(f"v40 numerical-resolution authorization is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != FINAL_STATUS:
        raise NumericalResolutionError("v40 numerical-resolution authorization is not final")
    criteria = value.get("criteria", {})
    if criteria.get("absolute_log_temperature_gradient_at_most") != RESOLUTION_THRESHOLD:
        raise NumericalResolutionError("authorized gradient threshold is not exactly 1e-6")
    required_true = (
        "approved_muben_source_hash_required", "deterministic_repeatability_required",
        "finite_strictly_positive_temperature_required", "inner_validation_only",
        "validation_nll_nonincrease_required",
    )
    if not all(criteria.get(name) is True for name in required_true):
        raise NumericalResolutionError("v40 authorization criteria are incomplete")
    if value.get("original_convergence_result_must_be_preserved") is not True:
        raise NumericalResolutionError("original convergence preservation is not authorized")
    return value, sha256_file(path)


def cell_key(fold: int, seed: int) -> str:
    if (fold, seed) not in AUTHORIZED_CELLS:
        raise NumericalResolutionError("cell is outside the exact v40 authorized set")
    return f"fold_{fold}_seed_{seed}"


def public_fit(value: dict[str, Any]) -> dict[str, Any]:
    return {name: value.get(name) for name in FIT_FIELDS}


def fits_repeat_exactly(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return public_fit(first) == public_fit(second) and first.get("loss_curve") == second.get("loss_curve")


def assess_fit(first: dict[str, Any], second: dict[str, Any], expected_source: str,
        actual_source: str) -> dict[str, Any]:
    temperature = float(first.get("temperature", float("nan")))
    before = float(first.get("validation_nll_before", float("nan")))
    after = float(first.get("validation_nll_after", float("nan")))
    gradient = float(first.get("final_abs_log_T_gradient", float("nan")))
    checks = {
        "approved_muben_source_hash_matches": actual_source == expected_source,
        "deterministic_repeatability_exact": fits_repeat_exactly(first, second),
        "finite_strictly_positive_temperature": math.isfinite(temperature) and temperature > 0,
        "validation_nll_finite_nonincrease": math.isfinite(before) and math.isfinite(after) and after <= before,
        "absolute_log_temperature_gradient_at_most_1e_6": math.isfinite(gradient) and gradient <= RESOLUTION_THRESHOLD,
        "fit_scope_inner_validation_only": first.get("fit_split") == "inner_validation_only",
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "original_converged": bool(first.get("converged")),
        "original_convergence_status": first.get("convergence_status"),
        "fit": public_fit(first),
    }


def validate_resolution_record(path: str | Path, *, branch: str, source_sha256: str,
        validation_artifact_sha256: str, fitted: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "passed_v40_numerical_resolution" or record.get("outer_test_accessed") is not False:
        raise NumericalResolutionError("v40 resolution record is not passed and pre-outer")
    authorization, authorization_sha256 = load_authorization()
    if record.get("authorization_sha256") != authorization_sha256:
        raise NumericalResolutionError("v40 resolution record authorization hash mismatch")
    if record.get("approved_module_sha256") != source_sha256 or source_sha256 != authorization.get("muben_source_sha256"):
        raise NumericalResolutionError("v40 resolution record MUBen hash mismatch")
    if branch not in ("control", "random2"):
        raise NumericalResolutionError("v40 resolution branch must be control or random2")
    branch_record = record.get("fits", {}).get(branch, {})
    if branch_record.get("validation_artifact_sha256") != validation_artifact_sha256:
        raise NumericalResolutionError("v40 resolution record validation-artifact hash mismatch")
    if branch_record.get("accepted") is not True or branch_record.get("fit") != public_fit(fitted):
        raise NumericalResolutionError("current MUBen fit does not match the authorized deterministic record")
    if branch_record.get("original_converged") != bool(fitted.get("converged")):
        raise NumericalResolutionError("original convergence result was not preserved")
    return sha256_file(path), branch_record
