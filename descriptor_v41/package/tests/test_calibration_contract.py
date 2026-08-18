from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from alignn_stage2.calibration_contract import (FittedTemperature, TemperatureScalingContractError,
    ValidatedTemperatureScalerMissing, BINARY_EQUIVALENCE_ATOL, BINARY_EQUIVALENCE_RTOL,
    FINAL_APPROVAL_STATUS, _load_approved_module, apply_temperature, fit_temperature,
    verify_binary_ordering_with_tolerance, verify_binary_shared_temperature_identity)


SYNTHETIC_LOGITS = np.asarray([[-2.0, 1.0], [3.0, -1.0], [0.2, 0.1], [-0.3, 0.8]], dtype=np.float64)
SYNTHETIC_LABELS = np.asarray([1, 0, 0, 1], dtype=np.int64)
TEST_DOUBLE = '''\nimport numpy as np\ndef fit_temperature(validation_logits, validation_labels):\n    return {"temperature": 1.0, "objective": "unweighted_validation_nll", "converged": True, "optimization_steps": 0, "validation_nll_before": 0.25, "validation_nll_after": 0.25, "implementation_version": "synthetic-contract-test-double"}\ndef apply_temperature(logits, fitted_object):\n    assert fitted_object["temperature"] == 1.0\n    return np.asarray(logits).copy()\n'''
SHARED_T_TEST_DOUBLE = '''\nimport numpy as np\ndef fit_temperature(validation_logits, validation_labels):\n    return {"temperature": 2.5, "objective": "unweighted_validation_nll", "converged": True, "optimization_steps": 3, "validation_nll_before": 0.5, "validation_nll_after": 0.4, "implementation_version": "synthetic-shared-t-contract-test-double"}\ndef apply_temperature(logits, fitted_object):\n    return np.asarray(logits) / fitted_object["temperature"]\n'''
INDEPENDENT_OUTPUT_TEST_DOUBLE = '''\nimport numpy as np\ndef fit_temperature(validation_logits, validation_labels):\n    return {"temperature": 2.5, "objective": "unweighted_validation_nll", "converged": True, "optimization_steps": 3, "validation_nll_before": 0.5, "validation_nll_after": 0.4, "implementation_version": "invalid-independent-output-test-double"}\ndef apply_temperature(logits, fitted_object):\n    return np.asarray(logits) / np.asarray([2.0, 3.0])\n'''


def approved_test_double(tmp_path: Path) -> Path:
    source = tmp_path / "synthetic_contract_test_double.py"
    source.write_text(TEST_DOUBLE, encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps({"status": FINAL_APPROVAL_STATUS, "installed_source": source.name, "expected_sha256": digest}), encoding="utf-8")
    return approval


def approved_source(tmp_path: Path, source_text: str, name: str) -> Path:
    source = tmp_path / name
    source.write_text(source_text, encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    approval = tmp_path / f"{name}.approval.json"
    approval.write_text(json.dumps({"status": FINAL_APPROVAL_STATUS, "installed_source": source.name, "expected_sha256": digest}), encoding="utf-8")
    return approval


def test_placeholder_fails_closed(tmp_path):
    with pytest.raises(ValidatedTemperatureScalerMissing):
        fit_temperature(SYNTHETIC_LOGITS, SYNTHETIC_LABELS, approval_path=tmp_path / "missing.json")


def test_pending_muben_approval_fails_closed(tmp_path):
    approval = approved_test_double(tmp_path)
    record = json.loads(approval.read_text(encoding="utf-8"))
    record["status"] = "validated_on_completed_muben_cells_pending_unimol_reconciliation"
    approval.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValidatedTemperatureScalerMissing, match="not final"):
        fit_temperature(SYNTHETIC_LOGITS, SYNTHETIC_LABELS, approval_path=approval)


def test_no_fitting_on_outer_test_even_if_module_is_missing(tmp_path):
    with pytest.raises(TemperatureScalingContractError, match="outer-test fitting is prohibited"):
        fit_temperature(SYNTHETIC_LOGITS, SYNTHETIC_LABELS, split="outer_test", approval_path=tmp_path / "missing.json")


def test_synthetic_contract_dimensions_positive_t_invariance_and_repeatability(tmp_path):
    approval = approved_test_double(tmp_path)
    fitted_1 = fit_temperature(SYNTHETIC_LOGITS, SYNTHETIC_LABELS, approval_path=approval)
    fitted_2 = fit_temperature(SYNTHETIC_LOGITS.copy(), SYNTHETIC_LABELS.copy(), approval_path=approval)
    assert np.isfinite(fitted_1.temperature) and fitted_1.temperature > 0
    assert fitted_1.public_metadata() == fitted_2.public_metadata()
    scaled_1 = apply_temperature(SYNTHETIC_LOGITS, fitted_1, approval_path=approval)
    scaled_2 = apply_temperature(SYNTHETIC_LOGITS, fitted_2, approval_path=approval)
    assert scaled_1.shape == SYNTHETIC_LOGITS.shape
    assert np.array_equal(scaled_1, SYNTHETIC_LOGITS)  # T=1 is exact.
    assert np.array_equal(scaled_1, scaled_2)
    assert np.array_equal(np.argmax(scaled_1, axis=1), np.argmax(SYNTHETIC_LOGITS, axis=1))
    assert np.array_equal(np.argsort(scaled_1[:, 1] - scaled_1[:, 0], kind="stable"),
        np.argsort(SYNTHETIC_LOGITS[:, 1] - SYNTHETIC_LOGITS[:, 0], kind="stable"))


def test_two_class_softmax_equals_sigmoid_of_shared_scaled_margin_numerically():
    logits = np.asarray([[-100.0, 100.0], [100.0, -100.0], [-2.5, 0.5], [0.25, 0.25], [7.0, 8.0]], dtype=np.float64)
    temperature = 2.5
    shifted = logits / temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    softmax_positive = exponentials[:, 1] / exponentials.sum(axis=1)
    margin = (logits[:, 1] - logits[:, 0]) / temperature
    sigmoid_margin = np.empty_like(margin)
    nonnegative = margin >= 0
    sigmoid_margin[nonnegative] = 1.0 / (1.0 + np.exp(-margin[nonnegative]))
    exp_margin = np.exp(margin[~nonnegative])
    sigmoid_margin[~nonnegative] = exp_margin / (1.0 + exp_margin)
    assert np.allclose(softmax_positive, sigmoid_margin, rtol=BINARY_EQUIVALENCE_RTOL, atol=BINARY_EQUIVALENCE_ATOL)
    verify_binary_shared_temperature_identity(logits, temperature)


def test_roundoff_scale_tie_reordering_is_not_a_material_rank_change():
    raw = np.asarray([[0.0, 0.0], [0.0, 5e-8], [0.0, 1.0]], dtype=np.float64)
    scaled_roundoff = np.asarray([[0.0, 2e-8], [0.0, 0.0], [0.0, 0.5]], dtype=np.float64)
    verify_binary_ordering_with_tolerance(raw, scaled_roundoff)


def test_material_binary_margin_inversion_is_rejected():
    raw = np.asarray([[0.0, 0.0], [0.0, 0.1]], dtype=np.float64)
    inverted = np.asarray([[0.0, 0.2], [0.0, 0.0]], dtype=np.float64)
    with pytest.raises(TemperatureScalingContractError, match="ordering changed beyond"):
        verify_binary_ordering_with_tolerance(raw, inverted)


def test_adapter_accepts_one_shared_positive_scalar_temperature(tmp_path):
    approval = approved_source(tmp_path, SHARED_T_TEST_DOUBLE, "shared_t.py")
    fitted = fit_temperature(SYNTHETIC_LOGITS, SYNTHETIC_LABELS, approval_path=approval)
    scaled = apply_temperature(SYNTHETIC_LOGITS, fitted, approval_path=approval)
    assert np.allclose(scaled, SYNTHETIC_LOGITS / 2.5, rtol=BINARY_EQUIVALENCE_RTOL, atol=BINARY_EQUIVALENCE_ATOL)


def test_adapter_rejects_two_independent_output_temperatures(tmp_path):
    approval = approved_source(tmp_path, INDEPENDENT_OUTPUT_TEST_DOUBLE, "independent_outputs.py")
    fitted = fit_temperature(SYNTHETIC_LOGITS, SYNTHETIC_LABELS, approval_path=approval)
    with pytest.raises(TemperatureScalingContractError, match="one shared positive scalar"):
        apply_temperature(SYNTHETIC_LOGITS, fitted, approval_path=approval)


def test_labels_and_logits_must_remain_aligned(tmp_path):
    approval = approved_test_double(tmp_path)
    with pytest.raises(TemperatureScalingContractError, match="aligned"):
        fit_temperature(SYNTHETIC_LOGITS, SYNTHETIC_LABELS[:-1], approval_path=approval)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_temperature_is_rejected(temperature):
    fitted = FittedTemperature(temperature, "nll", True, 1, 1.0, 0.9, "test", "0" * 64, None)
    with pytest.raises(TemperatureScalingContractError):
        fitted.validate()


def test_unapproved_module_hash_is_rejected(tmp_path):
    approval = approved_test_double(tmp_path)
    record = json.loads(approval.read_text(encoding="utf-8"))
    record["expected_sha256"] = "0" * 64
    approval.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValidatedTemperatureScalerMissing, match="SHA-256 mismatch"):
        _load_approved_module(approval)


def test_final_oof_analysis_refuses_missing_approval_before_reading_artifacts(tmp_path, monkeypatch):
    from alignn_stage2 import consolidate_oof
    monkeypatch.setattr(consolidate_oof, "DEFAULT_APPROVAL", tmp_path / "missing.json")
    with pytest.raises(ValidatedTemperatureScalerMissing):
        consolidate_oof.main(["--work-root", str(tmp_path / "absent"), "--output-dir", str(tmp_path / "output"), "--seeds", "0"])


def test_vector_temperature_is_rejected_explicitly():
    fitted = FittedTemperature(np.asarray([1.0, 1.0]), "nll", True, 1, 1.0, 0.9, "test", "0" * 64, None)
    with pytest.raises(TemperatureScalingContractError, match="exactly one scalar"):
        fitted.validate()


def test_real_vendored_source_hash_and_determinism():
    root = Path(__file__).resolve().parents[1]
    approval = root / "MUBEN_TS_APPROVAL.json"
    record = json.loads(approval.read_text(encoding="utf-8"))
    source = root / record["installed_source"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == record["expected_sha256"]
    first = fit_temperature(SYNTHETIC_LOGITS, SYNTHETIC_LABELS, approval_path=approval)
    second = fit_temperature(SYNTHETIC_LOGITS.copy(), SYNTHETIC_LABELS.copy(), approval_path=approval)
    assert first.public_metadata() == second.public_metadata()
    raw = SYNTHETIC_LOGITS.copy()
    scaled = apply_temperature(raw, first, approval_path=approval)
    assert np.array_equal(raw, SYNTHETIC_LOGITS)
    assert np.allclose(scaled, raw / first.temperature)
    assert first.implementation_object["parameterization"] == "T=exp(log_T)"
    assert first.implementation_object["numerical_dtype"] == "torch.float64"
    assert first.implementation_object["convergence_status"] == "converged"


def test_extreme_finite_logits_supported_without_probability_reconstruction():
    root = Path(__file__).resolve().parents[1]
    approval = root / "MUBEN_TS_APPROVAL.json"
    logits = np.asarray([[-1e6, 1e6], [1e6, -1e6], [-1e-12, 1e-12], [9e5, 9e5]], dtype=np.float64)
    labels = np.asarray([1, 0, 1, 0])
    fitted = fit_temperature(logits, labels, approval_path=approval)
    scaled = apply_temperature(logits, fitted, approval_path=approval)
    assert np.isfinite(scaled).all()
    assert np.array_equal(np.argmax(logits, axis=1), np.argmax(scaled, axis=1))


def test_control_random2_fold_seed_fit_isolation_is_explicit_in_export_source():
    source = (Path(__file__).resolve().parents[1] / "alignn_stage2" / "calibrate_export.py").read_text(encoding="utf-8")
    assert 'f"fold_{args.fold}" / f"seed_{args.seed}"' in source
    assert "fitted_control = fit_temperature(control[\"logits\"]" in source
    assert "fitted_random2 = fit_temperature(random2[\"logits\"]" in source
    assert '"fit_scope": "inner_validation_only"' in source
