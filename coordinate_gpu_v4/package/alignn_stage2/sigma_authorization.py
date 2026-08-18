"""Central fail-closed Random2-Coordinate sigma authorization boundary."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "COORDINATE_NOISE_CONFIG.json"
BINDING_PATH = PACKAGE_ROOT / "GPU_COORDINATE_V29_AUTHORIZATION_BINDING.json"


class CoordinateSigmaAuthorizationMissing(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class AuthorizedSigma:
    sigma: float
    authorization_path: str
    authorization_sha256: str
    config_template_sha256: str
    package_aggregate_sha256: str
    prospective_provenance_sha256: str
    test_only: bool = False


def _positive_finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CoordinateSigmaAuthorizationMissing("coordinate sigma must be numeric")
    sigma = float(value)
    if not math.isfinite(sigma) or sigma <= 0:
        raise CoordinateSigmaAuthorizationMissing("coordinate sigma must be finite and strictly positive")
    return sigma


def require_authorized_sigma(*, authorization_path: str | Path | None = None,
        test_config: dict | None = None, allow_test_only: bool = False) -> AuthorizedSigma:
    """Validate before any scientific labels, structures, cache, or outer split are read."""
    template = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    template_sha = sha256_file(CONFIG_PATH)
    if template.get("sigma_cartesian_per_axis_angstrom") is not None or template.get("sigma_status") != "pending_prospective_selection":
        raise CoordinateSigmaAuthorizationMissing("immutable release template must retain a null pending sigma")
    if test_config is not None:
        if not allow_test_only or os.getenv("ALIGNN_COORDINATE_TEST_MODE") != "1":
            raise CoordinateSigmaAuthorizationMissing("test-only sigma is prohibited outside explicit test mode")
        if test_config.get("test_only") is not True or test_config.get("scientific_use_prohibited") is not True:
            raise CoordinateSigmaAuthorizationMissing("synthetic sigma must be marked test_only and scientific_use_prohibited")
        sigma = _positive_finite(test_config.get("sigma_cartesian_per_axis_angstrom"))
        return AuthorizedSigma(sigma, "<test-only-memory>", hashlib.sha256(json.dumps(
            test_config, sort_keys=True).encode()).hexdigest(), template_sha, "test-only", "test-only", True)
    path = Path(authorization_path or os.getenv("ALIGNN_COORDINATE_SIGMA_AUTHORIZATION", ""))
    if not str(path) or not path.is_file():
        raise CoordinateSigmaAuthorizationMissing(
            "pending_coordinate_sigma_authorization: set ALIGNN_COORDINATE_SIGMA_AUTHORIZATION to a verified prospective authority artifact")
    value = json.loads(path.read_text(encoding="utf-8"))
    sigma = _positive_finite(value.get("sigma_cartesian_per_axis_angstrom"))
    binding = json.loads(BINDING_PATH.read_text(encoding="utf-8"))
    authorization_sha = sha256_file(path)
    checks = {
        "status": value.get("status") == "approved_prospective_coordinate_sigma",
        "authorized": value.get("scientific_execution_authorized") is True,
        "prospective": value.get("selected_prospectively_without_alignn_outcomes") is True,
        "test_only_false": value.get("test_only") is False,
        "template_hash": value.get("coordinate_noise_config_template_sha256") == template_sha,
        "origin_release": value.get("release") == binding.get("authorization_origin_release"),
        "origin_package_hash": value.get("package_aggregate_sha256")
            == binding.get("authorization_origin_package_aggregate_sha256"),
        "authorization_hash": authorization_sha == binding.get("authorization_sha256"),
        "bound_template_hash": template_sha == binding.get("coordinate_noise_config_template_sha256"),
        "bound_sigma": sigma == binding.get("sigma_cartesian_per_axis_angstrom"),
        "selection_route": value.get("sigma_selection_route") == binding.get("sigma_selection_route"),
        "provenance_hash": isinstance(value.get("prospective_provenance_sha256"), str)
            and len(value["prospective_provenance_sha256"]) == 64,
    }
    if not all(checks.values()):
        raise CoordinateSigmaAuthorizationMissing(f"coordinate sigma authorization rejected: {checks}")
    return AuthorizedSigma(sigma, str(path.resolve()), authorization_sha, template_sha,
        value["package_aggregate_sha256"], value["prospective_provenance_sha256"], False)
