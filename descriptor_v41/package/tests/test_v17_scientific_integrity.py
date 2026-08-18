from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from alignn_stage2.calibrate_export import control_adequacy, validate_pair
from alignn_stage2.consolidate_oof import clustered_structure_bootstrap
from alignn_stage2.execution_contracts import run_execution_contracts
from alignn_stage2.production import (ECE_BIN_COUNT, adaptive_reliability_bins,
    classwise_calibration_error, metrics, reliability_bins)
from alignn_stage2.training import (assert_native_logit_contract, component_state, forward_raw,
    states_byte_identical, validation_nll)

ROOT = Path(__file__).resolve().parents[1]


class FakeAlignnClassification(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 2)

    def forward(self, values):
        # Reproduce native ALIGNN's singleton squeeze behavior.
        return torch.nn.functional.log_softmax(self.fc(values).squeeze(), dim=-1)


def test_pre_logsoftmax_fc_output_and_loss_identity_with_final_batch_one():
    model = FakeAlignnClassification()
    labels = torch.tensor([1])
    log_probabilities, logits = forward_raw(model, torch.ones((1, 3)), labels)
    assert log_probabilities.shape == logits.shape == (1, 2)
    assert_native_logit_contract(log_probabilities, logits, labels)
    assert torch.allclose(torch.nn.functional.nll_loss(log_probabilities, labels),
        torch.nn.functional.cross_entropy(logits, labels), rtol=1e-6, atol=1e-7)


def test_complete_non_head_state_comparison_includes_batchnorm_buffers():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 2))
    before = component_state(model.state_dict(), classifier=False)
    after = {name: value.clone() for name, value in before.items()}
    tracked = next(name for name in after if name.endswith("num_batches_tracked"))
    after[tracked] += 1
    assert not states_byte_identical(before, after)


def test_validation_artifact_requires_exact_z0_z1_pre_logsoftmax_provenance(tmp_path):
    common = dict(structure_ids=np.asarray(["a", "b"]), labels=np.asarray([0, 1]),
        sample_order_index=np.asarray([0, 1]), logits=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        logit_source=np.asarray("ALIGNN.fc output before LogSoftmax"), logit_columns=np.asarray(["z_0", "z_1"]))
    left, right = tmp_path / "left.npz", tmp_path / "right.npz"
    np.savez(left, **common); np.savez(right, **common)
    with np.load(left, allow_pickle=False) as control, np.load(right, allow_pickle=False) as random2:
        validate_pair(control, random2)
    common["logit_source"] = np.asarray("post-LogSoftmax log-probabilities")
    np.savez(right, **common)
    with np.load(left, allow_pickle=False) as control, np.load(right, allow_pickle=False) as random2:
        with pytest.raises(RuntimeError, match="pre-LogSoftmax"):
            validate_pair(control, random2)


def test_validation_nll_is_sample_weighted_not_mean_of_batch_means():
    logits = np.asarray([[4.0, -4.0], [4.0, -4.0], [-4.0, 4.0]], dtype=np.float32)
    labels = [0, 0, 0]
    combined = validation_nll(logits, labels)
    batch_2 = validation_nll(logits[:2], labels[:2])
    batch_1 = validation_nll(logits[2:], labels[2:])
    assert combined == pytest.approx((2 * batch_2 + batch_1) / 3)
    assert combined != pytest.approx((batch_2 + batch_1) / 2)


def test_metric_definitions_boundaries_adaptive_classwise_brier_and_auc():
    labels = np.asarray([0, 1, 1, 0, 1, 0])
    probabilities = np.asarray([0.0, 0.5, 8 / 15, 1.0, 0.8, 0.2])
    bins = reliability_bins(labels, probabilities)
    assert len(bins) == ECE_BIN_COUNT and sum(row["count"] for row in bins) == len(labels)
    adaptive = adaptive_reliability_bins(labels, probabilities)
    assert len(adaptive) == ECE_BIN_COUNT and sum(row["count"] for row in adaptive) == len(labels)
    classwise = classwise_calibration_error(labels, probabilities)
    assert set(classwise["per_class"]) == {"0", "1"}
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    logits = np.column_stack((np.zeros(len(labels)), np.log(clipped / (1 - clipped))))
    result = metrics(labels, logits)
    assert result["brier_positive_class"] == pytest.approx(np.mean((clipped - labels) ** 2))
    assert result["metric_definitions"]["roc_auc"].startswith("native binary logit margin")
    assert "adaptive_ece_15_top_label" in result and "classwise_ece_15_macro" in result


def test_clustered_bootstrap_carries_all_seeds_for_each_sampled_structure():
    values = np.asarray([[1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5]])
    result = clustered_structure_bootstrap(values, repetitions=200, seed=77)
    assert result["estimate"] == pytest.approx(np.mean(values.mean(axis=1)))
    assert result["per_seed_estimates"] == pytest.approx(values.mean(axis=1))
    identical = clustered_structure_bootstrap(np.ones((5, 8)) * 0.25, repetitions=50, seed=5)
    assert identical["ci95_low"] == pytest.approx(0.25)
    assert identical["ci95_high"] == pytest.approx(0.25)
    assert identical["unit"] == "structure_cluster_with_all_seed_predictions"


def test_validation_adequacy_gate_is_outcome_neutral_to_random2():
    source = (ROOT / "alignn_stage2" / "calibrate_export.py").read_text(encoding="utf-8")
    function = source[source.index("def control_adequacy"):source.index("def load_outer_test")]
    assert "def control_adequacy(logits: np.ndarray, labels: np.ndarray, train_prior: float)" in function
    assert "random2_logits" not in function and "random2_labels" not in function
    logits = np.asarray([[3.0, -1.0], [-1.0, 3.0], [2.0, -0.5], [-0.5, 2.0]])
    result = control_adequacy(logits, np.asarray([0, 1, 0, 1]), 0.5)
    assert result["outcome_neutral_with_respect_to_random2"]
    assert not result["random2_comparison_used_for_gate"]


def test_synthetic_execution_resume_onecycle_rng_initial_hash_and_invariance():
    result = run_execution_contracts("cpu")
    assert result["status"] == "passed"
    assert result["uninterrupted_model_sha256"] == result["resumed_model_sha256"]
    assert result["optimizer_steps"] == result["onecycle_scheduler_steps"] == 6
    assert result["final_batch_size_one_exercised"]
    assert result["batchnorm_state_byte_identical_in_eval"]
    assert result["temperature_invariance"]["roc_auc_unchanged"]
