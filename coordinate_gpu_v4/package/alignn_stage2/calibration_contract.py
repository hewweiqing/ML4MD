"""Fail-closed adapter for the future validated MUBen temperature scaler.

This module contains no temperature-fitting algorithm. It validates and calls an
installed module only after its source digest exactly matches the approved digest.
"""

from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .common import sha256_file

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APPROVAL = PACKAGE_ROOT / "MUBEN_TS_APPROVAL.json"
FINAL_APPROVAL_STATUS = "approved_muben_v31_numerical_convergence_amendment"
BINARY_EQUIVALENCE_ATOL = 1e-7
BINARY_EQUIVALENCE_RTOL = 1e-6


class ValidatedTemperatureScalerMissing(RuntimeError):
    """Raised before scientific calibration/test access when no approved scaler exists."""


class TemperatureScalingContractError(RuntimeError):
    """Raised when an installed scaler violates the frozen adapter contract."""


@dataclass(frozen=True)
class FittedTemperature:
    temperature: float
    objective: str
    converged: bool
    optimization_steps: int
    validation_nll_before: float
    validation_nll_after: float
    implementation_version: str
    source_sha256: str
    implementation_object: Any

    def validate(self) -> "FittedTemperature":
        temperature_array = np.asarray(self.temperature)
        if temperature_array.ndim != 0:
            raise TemperatureScalingContractError(
                "fitted T must be exactly one scalar; vector, per-column, and independent-output temperatures are prohibited"
            )
        if not math.isfinite(float(temperature_array)) or float(temperature_array) <= 0:
            raise TemperatureScalingContractError("fitted T must be finite and strictly positive")
        if self.optimization_steps < 0:
            raise TemperatureScalingContractError("optimization_steps must be non-negative")
        if not isinstance(self.converged, (bool, np.bool_)) or not bool(self.converged):
            raise TemperatureScalingContractError("temperature fitting must report successful convergence")
        if "nll" not in self.objective.lower():
            raise TemperatureScalingContractError("fitting objective must be validation NLL")
        for name in ("validation_nll_before", "validation_nll_after"):
            if not math.isfinite(float(getattr(self, name))):
                raise TemperatureScalingContractError(f"{name} must be finite")
        if not self.objective or not self.implementation_version or len(self.source_sha256) != 64:
            raise TemperatureScalingContractError("incomplete fitted-temperature provenance")
        return self

    def public_metadata(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("implementation_object")
        return value


def _binary_logits(logits: Any) -> np.ndarray:
    array = np.asarray(logits)
    if array.ndim != 2 or array.shape[1] != 2:
        raise TemperatureScalingContractError("native binary logits must have shape [n, 2]")
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise TemperatureScalingContractError("native binary logits must be finite floating-point values")
    return array


def _labels(labels: Any, n: int) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 1 or len(array) != n or not np.isin(array, [0, 1]).all():
        raise TemperatureScalingContractError("binary labels must be a length-n vector aligned to logits")
    return array.astype(np.int64, copy=False)


def _two_class_positive_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials[:, 1] / np.sum(exponentials, axis=1)


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.result_type(values, np.float64))
    nonnegative = values >= 0
    output[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    output[~nonnegative] = exponential / (1.0 + exponential)
    return output


def verify_binary_shared_temperature_identity(logits: Any, temperature: float) -> None:
    raw = _binary_logits(logits)
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise TemperatureScalingContractError("shared binary temperature must be finite and strictly positive")
    softmax_positive = _two_class_positive_softmax(raw / float(temperature))
    sigmoid_margin = _stable_sigmoid((raw[:, 1] - raw[:, 0]) / float(temperature))
    if not np.allclose(softmax_positive, sigmoid_margin, rtol=BINARY_EQUIVALENCE_RTOL, atol=BINARY_EQUIVALENCE_ATOL):
        difference = float(np.max(np.abs(softmax_positive - sigmoid_margin)))
        raise TemperatureScalingContractError(
            f"binary softmax/sigmoid identity failed (max_abs_diff={difference}, "
            f"atol={BINARY_EQUIVALENCE_ATOL}, rtol={BINARY_EQUIVALENCE_RTOL})"
        )


def verify_binary_ordering_with_tolerance(raw: Any, scaled: Any) -> None:
    """Reject material margin inversions while treating roundoff-scale ties as ties."""
    raw_array, scaled_array = _binary_logits(raw), _binary_logits(scaled)
    if raw_array.shape != scaled_array.shape:
        raise TemperatureScalingContractError("binary raw/scaled logit dimensions differ")
    raw_margin = raw_array[:, 1] - raw_array[:, 0]
    scaled_margin = scaled_array[:, 1] - scaled_array[:, 0]
    raw_order = np.argsort(raw_margin, kind="stable")
    ordered = scaled_margin[raw_order]
    if len(ordered) < 2:
        return
    differences = np.diff(ordered)
    adjacent_scale = np.maximum(np.abs(ordered[:-1]), np.abs(ordered[1:]))
    allowance = BINARY_EQUIVALENCE_ATOL + BINARY_EQUIVALENCE_RTOL * adjacent_scale
    if np.any(differences < -allowance):
        worst = float(np.min(differences + allowance))
        raise TemperatureScalingContractError(
            f"binary raw-logit ordering changed beyond declared numerical tolerance (worst={worst})"
        )


def _load_approved_module(approval_path: str | Path = DEFAULT_APPROVAL) -> tuple[ModuleType, str]:
    approval_path = Path(approval_path)
    if not approval_path.is_file():
        raise ValidatedTemperatureScalerMissing(f"MUBen TS approval record is missing: {approval_path}")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval.get("status") != FINAL_APPROVAL_STATUS:
        raise ValidatedTemperatureScalerMissing("MUBen TS approval is not final; pending and generic approval states fail closed")
    expected = approval.get("expected_sha256")
    source_value = approval.get("installed_source")
    if not expected or len(expected) != 64 or not source_value:
        raise ValidatedTemperatureScalerMissing("validated MUBen TS source and expected SHA-256 are not installed")
    source = (approval_path.parent / source_value).resolve()
    if not source.is_file():
        raise ValidatedTemperatureScalerMissing(f"approved MUBen TS source is absent: {source}")
    actual = sha256_file(source)
    if actual != expected:
        raise ValidatedTemperatureScalerMissing(f"MUBen TS SHA-256 mismatch: expected {expected}, found {actual}")
    spec = importlib.util.spec_from_file_location("validated_muben_temperature_scaling", source)
    if spec is None or spec.loader is None:
        raise ValidatedTemperatureScalerMissing("cannot import approved MUBen TS source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "fit_temperature", None)) or not callable(getattr(module, "apply_temperature", None)):
        raise TemperatureScalingContractError("approved module must export fit_temperature and apply_temperature")
    declared_version = approval.get("implementation_version")
    module_version = getattr(module, "IMPLEMENTATION_VERSION", declared_version)
    if declared_version and module_version != declared_version:
        raise TemperatureScalingContractError("approved implementation version does not match source")
    return module, actual


def fit_temperature(validation_logits: Any, validation_labels: Any, *, split: str = "validation", approval_path: str | Path = DEFAULT_APPROVAL) -> FittedTemperature:
    if split != "validation":
        raise TemperatureScalingContractError("temperature fitting is permitted on inner validation only; outer-test fitting is prohibited")
    logits = _binary_logits(validation_logits)
    labels = _labels(validation_labels, len(logits))
    module, digest = _load_approved_module(approval_path)
    fitted = module.fit_temperature(logits, labels)
    required = {
        "temperature", "objective", "converged", "optimization_steps",
        "validation_nll_before", "validation_nll_after", "implementation_version",
    }
    if isinstance(fitted, dict):
        missing = required - set(fitted)
        if missing:
            raise TemperatureScalingContractError(f"fitted object missing fields: {sorted(missing)}")
        result = FittedTemperature(source_sha256=digest, implementation_object=fitted, **{key: fitted[key] for key in required})
    else:
        missing = [key for key in required if not hasattr(fitted, key)]
        if missing:
            raise TemperatureScalingContractError(f"fitted object missing attributes: {missing}")
        result = FittedTemperature(source_sha256=digest, implementation_object=fitted, **{key: getattr(fitted, key) for key in required})
    return result.validate()


def apply_temperature(logits: Any, fitted: FittedTemperature, *, approval_path: str | Path = DEFAULT_APPROVAL) -> np.ndarray:
    raw = _binary_logits(logits)
    fitted.validate()
    module, digest = _load_approved_module(approval_path)
    if digest != fitted.source_sha256:
        raise TemperatureScalingContractError("fitted object source hash does not match the installed approved module")
    scaled = np.asarray(module.apply_temperature(raw, fitted.implementation_object))
    if scaled.shape != raw.shape or not np.isfinite(scaled).all():
        raise TemperatureScalingContractError("scaled logits must be finite and preserve dimensions")
    expected_shared_scaling = raw / fitted.temperature
    if not np.allclose(scaled, expected_shared_scaling, rtol=BINARY_EQUIVALENCE_RTOL, atol=BINARY_EQUIVALENCE_ATOL):
        raise TemperatureScalingContractError(
            "MUBen adapter must apply one shared positive scalar T to the binary logit matrix; "
            "two independent output temperatures are prohibited"
        )
    verify_binary_shared_temperature_identity(raw, fitted.temperature)
    if not np.array_equal(np.argmax(raw, axis=1), np.argmax(scaled, axis=1)):
        raise TemperatureScalingContractError("positive scalar scaling changed predicted labels")
    verify_binary_ordering_with_tolerance(raw, scaled)
    if fitted.temperature == 1.0 and not np.array_equal(raw, scaled):
        raise TemperatureScalingContractError("T=1 must leave logits exactly unchanged")
    return scaled
