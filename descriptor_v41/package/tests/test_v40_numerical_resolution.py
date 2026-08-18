from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from alignn_stage2.calibration_contract import FittedTemperature
from alignn_stage2.numerical_resolution import (
    AUTHORIZED_CELLS, RESOLUTION_THRESHOLD, NumericalResolutionError,
    assess_fit, cell_key, load_authorization, validate_resolution_record,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "108b3183400725ca53e025c6f0f92690f35069f6fd786091b9d37dc50e90f08b"


def fit(*, gradient=2.4e-7, temperature=1.3, before=0.25, after=0.24, converged=False):
    return {
        "temperature": temperature,
        "objective": "unweighted_inner_validation_nll_two_class_cross_entropy",
        "converged": converged,
        "convergence_status": "converged" if converged else "not_converged",
        "optimization_steps": 5,
        "validation_nll_before": before,
        "validation_nll_after": after,
        "final_abs_log_T_gradient": gradient,
        "implementation_version": "muben-final-446471d-alignn-logt-float64-lbfgs-v2-grad1e-7",
        "parameterization": "T=exp(log_T)",
        "numerical_dtype": "torch.float64",
        "optimizer": "LBFGS_strong_wolfe",
        "max_iterations": 500,
        "tolerance_grad": 1e-7,
        "tolerance_change": 1e-12,
        "fit_split": "inner_validation_only",
        "loss_curve": [before, after],
    }


def test_authorization_is_exact_and_honestly_post_training_pre_outer():
    value, digest = load_authorization()
    assert len(digest) == 64
    assert AUTHORIZED_CELLS == {(1, 1), (2, 4), (3, 1)}
    assert RESOLUTION_THRESHOLD == 1e-6
    assert value["decision_basis"] == "optimizer diagnostics only; no outer-test information"
    assert value["original_convergence_result_must_be_preserved"] is True
    try:
        cell_key(0, 0)
    except NumericalResolutionError:
        pass
    else:
        raise AssertionError("unauthorized cell was accepted")


def test_assessment_accepts_authorized_stationarity_but_preserves_false_convergence():
    original = fit()
    result = assess_fit(original, dict(original), SOURCE, SOURCE)
    assert result["accepted"] is True
    assert result["original_converged"] is False
    assert result["original_convergence_status"] == "not_converged"


def test_assessment_fails_closed_on_each_required_condition():
    base = fit()
    cases = [
        (fit(gradient=1.0000001e-6), dict(base), SOURCE, SOURCE),
        (fit(temperature=0.0), fit(temperature=0.0), SOURCE, SOURCE),
        (fit(after=0.26), fit(after=0.26), SOURCE, SOURCE),
        (base, dict(base, temperature=1.31), SOURCE, SOURCE),
        (base, dict(base), SOURCE, "0" * 64),
    ]
    for first, second, expected, actual in cases:
        assert assess_fit(first, second, expected, actual)["accepted"] is False


def test_resolution_record_binds_fit_module_authorization_and_artifact(tmp_path):
    authorization, authorization_sha = load_authorization()
    current = fit()
    assessed = assess_fit(current, dict(current), SOURCE, SOURCE)
    assessed["validation_artifact_sha256"] = "a" * 64
    record = {
        "status": "passed_v40_numerical_resolution",
        "outer_test_accessed": False,
        "authorization_sha256": authorization_sha,
        "approved_module_sha256": authorization["muben_source_sha256"],
        "fits": {"control": assessed},
    }
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    digest, branch = validate_resolution_record(path, branch="control", source_sha256=SOURCE,
        validation_artifact_sha256="a" * 64, fitted=current)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert branch["original_converged"] is False
    record["outer_test_accessed"] = True
    path.write_text(json.dumps(record), encoding="utf-8")
    try:
        validate_resolution_record(path, branch="control", source_sha256=SOURCE,
            validation_artifact_sha256="a" * 64, fitted=current)
    except NumericalResolutionError:
        pass
    else:
        raise AssertionError("post-outer resolution record was accepted")


def test_fitted_object_preserves_original_nonconvergence_metadata():
    fitted = FittedTemperature(temperature=1.3, objective="validation_nll", converged=False,
        optimization_steps=5, validation_nll_before=0.25, validation_nll_after=0.24,
        implementation_version="approved", source_sha256="1" * 64, implementation_object=fit(),
        convergence_status="not_converged", numerical_resolution_authorized=True,
        numerical_resolution_record_sha256="2" * 64)
    assert fitted.validate().public_metadata()["converged"] is False


def test_v40_validator_is_validation_only_and_export_order_is_fail_closed():
    validator = (ROOT / "scripts/authorize_v40_numerical_resolution.py").read_text(encoding="utf-8")
    tree = ast.parse(validator)
    names = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "load_outer_test" not in names and "calibrate_and_export" not in validator
    assert "OUTER_TEST_ACCESS_STARTED.json" in validator
    slurm = (ROOT / "slurm/13_resolve_export_preouter_v40.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --array=6,14,16%3" in slurm
    assert slurm.index("authorize_v40_numerical_resolution.py") < slurm.index("calibrate_and_export.py")
    assert slurm.index("calibrate_and_export.py") < slurm.index("verify_cell.py") < slurm.index("promote_cell.py")
