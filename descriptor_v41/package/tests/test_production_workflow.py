import json
from pathlib import Path

import numpy as np
import pytest

from alignn_stage2.production import (array_cell, assert_temperature_invariance, atomic_promote, metrics,
    reliability_bins, verify_complete, write_complete)


def test_array_mapping_is_exact_grid():
    assert [array_cell(i) for i in range(25)] == [(f, s) for f in range(5) for s in range(5)]
    with pytest.raises(ValueError):
        array_cell(25)


def test_metrics_reliability_and_positive_temperature_invariance():
    labels = np.array([0, 1, 0, 1, 1, 0])
    logits = np.array([[2., -1.], [-2., 3.], [1., 0.], [-1., 2.], [0., 1.], [3., -2.]])
    scaled = logits / 2.75
    result = metrics(labels, logits)
    assert set(("ece_15", "nll", "brier", "accuracy", "f1", "roc_auc", "mean_confidence",
                "class_balance_positive", "predicted_positive_rate", "reliability_bins")) <= result.keys()
    assert len(result["reliability_bins"]) == 15
    assert sum(row["count"] for row in reliability_bins(labels, result_probabilities(logits))) == len(labels)
    assert assert_temperature_invariance(logits, scaled)["ranking_unchanged"]


def test_production_invariance_accepts_only_roundoff_scale_tie_reordering():
    raw_margin = np.array([1.0e-5, 1.005e-5, 1.0, 2.0])
    raw = np.column_stack((np.zeros_like(raw_margin), raw_margin))
    scaled_margin = np.array([0.5002e-5, 0.5001e-5, 0.5, 1.0])
    scaled = np.column_stack((np.zeros_like(scaled_margin), scaled_margin))
    result = assert_temperature_invariance(raw, scaled)
    assert result["ranking_unchanged"]
    assert result["tie_pattern_unchanged"]


def test_production_invariance_rejects_material_margin_inversion():
    raw_margin = np.array([0.2, 0.8, 1.4])
    raw = np.column_stack((np.zeros_like(raw_margin), raw_margin))
    inverted_margin = np.array([0.2, 1.4, 0.8])
    inverted = np.column_stack((np.zeros_like(inverted_margin), inverted_margin))
    with pytest.raises(RuntimeError, match="temperature invariance failed"):
        assert_temperature_invariance(raw, inverted)


def result_probabilities(logits):
    margin = logits[:, 1] - logits[:, 0]
    return 1 / (1 + np.exp(-margin))


def test_complete_marker_is_hash_verified_and_non_vacuous(tmp_path: Path):
    required = ["control/best.pt", "random2/best.pt", "control/validation_raw_logits.npz",
        "random2/validation_raw_logits.npz", "predictions/outer_test_predictions.csv",
        "predictions/scaled_logits_sidecar.npz", "temperature_scaling_provenance.json",
        "split_provenance.json", "random2_diagnostics.json", "CELL_VERIFICATION.json"]
    for index, name in enumerate(required):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"artifact-{index}".encode())
    assert not verify_complete(tmp_path, 0, 0)
    write_complete(tmp_path, 0, 0, "a" * 64)
    assert verify_complete(tmp_path, 0, 0)
    (tmp_path / required[0]).write_bytes(b"tampered")
    assert not verify_complete(tmp_path, 0, 0)


def test_atomic_promotion_moves_verified_staging_tree(tmp_path: Path):
    staging, final, quarantine = tmp_path / "staging", tmp_path / "final", tmp_path / "quarantine"
    staging.mkdir(); (staging / "evidence.txt").write_text("persistent", encoding="utf-8")
    atomic_promote(staging, final, quarantine)
    assert not staging.exists()
    assert (final / "evidence.txt").read_text(encoding="utf-8") == "persistent"
