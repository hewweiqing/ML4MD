import pytest
import torch

from alignn_stage2 import coordinate_head_warmup, descriptor_head_warmup
from alignn_stage2.coordinate_head_warmup import warm_coordinate_head
from alignn_stage2.coordinate_noise import balanced_random_labels
from alignn_stage2.descriptor_head_warmup import warm_descriptor_head


@pytest.fixture(autouse=True)
def _fast_budget(monkeypatch):
    # Real budget is 938 steps/3000 synthetic samples; reduced here purely
    # for test wall-clock speed, not to change the mechanism.
    monkeypatch.setattr(descriptor_head_warmup, "WARMUP_STEPS", 10)
    monkeypatch.setattr(descriptor_head_warmup, "SYNTHETIC_COUNT", 128)
    monkeypatch.setattr(coordinate_head_warmup, "WARMUP_STEPS", 10)
    monkeypatch.setattr(coordinate_head_warmup, "RECORD_COUNT", 128)


def _fresh_fc_state(dim: int = 6) -> dict:
    return torch.nn.Linear(dim, 2).state_dict()


def test_descriptor_head_warmup_deterministic_and_changes_head():
    torch.manual_seed(0)
    descriptors = torch.randn(200, 6)
    fc_state = _fresh_fc_state()
    result = warm_descriptor_head(fc_state, descriptors, seed=5)
    assert result["deterministic_replay"] is True
    assert not torch.equal(result["fc_state"]["weight"], fc_state["weight"])
    assert result["diagnostics"]["finite"] is True


def test_coordinate_head_warmup_deterministic_and_requires_balanced_labels():
    torch.manual_seed(1)
    descriptors = torch.randn(128, 6)
    labels = torch.as_tensor(balanced_random_labels(128, seed=17), dtype=torch.long)
    fc_state = _fresh_fc_state()
    result = warm_coordinate_head(fc_state, descriptors, labels, version=1, fold=0, seed=3)
    assert result["deterministic_replay"] is True
    assert not torch.equal(result["fc_state"]["weight"], fc_state["weight"])


def test_coordinate_head_warmup_rejects_unbalanced_labels():
    descriptors = torch.randn(128, 6)
    unbalanced = torch.zeros(128, dtype=torch.long)
    with pytest.raises(RuntimeError, match="balanced"):
        warm_coordinate_head(_fresh_fc_state(), descriptors, unbalanced, version=1, fold=0, seed=3)


def test_coordinate_head_warmup_rejects_wrong_record_count():
    descriptors = torch.randn(64, 6)
    labels = torch.as_tensor(balanced_random_labels(64, seed=1), dtype=torch.long)
    with pytest.raises(RuntimeError, match="128"):
        warm_coordinate_head(_fresh_fc_state(), descriptors, labels, version=1, fold=0, seed=3)
