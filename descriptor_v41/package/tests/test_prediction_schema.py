from __future__ import annotations

import pytest

from alignn_stage2.prediction_schema import PredictionSchemaError, stable_positive_probability, validate_rows


def row(index=0):
    return {"structure_id": f"synthetic-{index}", "fold": 0, "seed": 0, "condition": "A_Control_raw",
        "split": "inner_validation", "true_label": index % 2, "raw_native_logit_0": -0.25,
        "raw_native_logit_1": 0.75, "raw_probability_positive_audit_only": stable_positive_probability(-0.25, 0.75),
        "predicted_label": 1, "checkpoint_sha256": "a" * 64, "sample_order_index": index}


def test_schema_accepts_aligned_synthetic_rows():
    assert len(validate_rows([row(0), row(1)], expected_split="inner_validation")) == 2


def test_schema_rejects_misaligned_predicted_label():
    value = row()
    value["predicted_label"] = 0
    with pytest.raises(PredictionSchemaError, match="does not match"):
        validate_rows([value])


def test_schema_rejects_duplicate_structure_condition_key():
    with pytest.raises(PredictionSchemaError, match="duplicate"):
        validate_rows([row(), row()])

