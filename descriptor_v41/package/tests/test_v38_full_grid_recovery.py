from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from alignn_stage2.production import assert_temperature_invariance, metrics

ROOT = Path(__file__).resolve().parents[1]


def test_probability_saturation_does_not_invalidate_native_margin_auc_invariance():
    labels = np.asarray([0, 1, 0, 1])
    margins = np.asarray([1000.0, 1003.0, 1001.0, 1004.0])
    raw = np.column_stack((np.zeros(4), margins))
    scaled = raw / 2.0
    result = assert_temperature_invariance(raw, scaled, labels)
    assert result["roc_auc_unchanged"] is True
    assert metrics(labels, raw)["roc_auc"] == metrics(labels, scaled)["roc_auc"]


def test_material_margin_inversion_still_fails_closed():
    raw = np.column_stack((np.zeros(4), [0.0, 1.0, 2.0, 3.0]))
    changed = np.column_stack((np.zeros(4), [0.0, 2.0, 1.0, 3.0]))
    try:
        assert_temperature_invariance(raw, changed, np.asarray([0, 1, 0, 1]))
    except RuntimeError:
        pass
    else:
        raise AssertionError("material ordering inversion was accepted")


def test_existing_export_recovery_is_result_blind_and_never_rematerializes():
    path = ROOT / "scripts/recover_v38_existing_export.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "metrics" not in called
    assert "fit_temperature" not in called
    assert "load_outer_test" not in called
    assert "calibrate_and_export" not in source
    assert '["true_label"]' not in source and "['true_label']" not in source
    assert "outer_test_metrics_computed_or_compared\": False" in source
    assert "outer_test_rematerialized\": False" in source


def test_recovery_sets_are_exact_and_disjoint():
    existing = (ROOT / "scripts/recover_v38_existing_export.py").read_text(encoding="utf-8")
    diagnostic = (ROOT / "scripts/diagnose_v38_preouter_convergence.py").read_text(encoding="utf-8")
    assert "{(1, 4), (2, 0), (3, 0), (3, 2), (3, 4)}" in existing
    assert "{(1, 1): \"control\", (2, 4): \"random2\", (3, 1): \"control\"}" in diagnostic
    assert "OUTER_TEST_ACCESS_STARTED.json" in both_sources(existing, diagnostic)[0]
    assert "outer-test sentinel exists" in diagnostic


def both_sources(left: str, right: str):
    return left, right


def test_preouter_diagnostic_cannot_authorize_a_new_algorithm_or_outer_access():
    source = (ROOT / "scripts/diagnose_v38_preouter_convergence.py").read_text(encoding="utf-8")
    assert "module.fit_temperature" in source
    assert "blocked_pending_validated_muben_numerical_resolution" in source
    assert "No tolerance relaxation, optimizer substitution, or alternative fitter" in source
    assert "load_outer_test" not in source
    assert "outer_test_accessed\": False" in source


def test_slurm_arrays_match_observed_v37_failures():
    diagnostic = (ROOT / "slurm/11_diagnose_preouter_convergence_v38.sbatch").read_text(encoding="utf-8")
    existing = (ROOT / "slurm/12_recover_existing_exports_v38.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --array=6,14,16%3" in diagnostic
    assert "#SBATCH --array=9,10,15,17,19%5" in existing
    assert "calibrate_and_export.py" not in existing
    assert "promote_cell.py" in existing
