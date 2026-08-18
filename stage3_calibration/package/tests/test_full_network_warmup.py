import math

import pytest
import torch

from alignn_stage2 import full_network_warmup
from alignn_stage2.activation_probe import head_displacement_norm
from alignn_stage2.full_network_warmup import warm_full_network


class _SyntheticFullModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.BatchNorm1d(8), torch.nn.ReLU())
        self.fc = torch.nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.encoder(x))


def _model_factory() -> torch.nn.Module:
    return _SyntheticFullModel()


def _forward_logits(model: torch.nn.Module, batch_input) -> torch.Tensor:
    return model(batch_input)


def _pure_batch_factory(step: int, seed: int):
    generator = torch.Generator().manual_seed(seed * 100000 + step)
    inputs = torch.randn(8, 5, generator=generator)
    labels = torch.randint(0, 2, (8,), generator=generator)
    return inputs, labels


@pytest.fixture(autouse=True)
def _fast_warmup_budget(monkeypatch):
    # Real budget is 938 steps (see FULL_NETWORK_WARMUP_CONFIG.json); reduced
    # here purely for test wall-clock speed, not to change the mechanism.
    monkeypatch.setattr(full_network_warmup, "WARMUP_STEPS", 15)


def test_warm_full_network_deterministic_replay_and_encoder_changed():
    torch.manual_seed(0)
    initial_state = _model_factory().state_dict()
    result = warm_full_network(_model_factory, initial_state, _pure_batch_factory, _forward_logits, seed=11)
    assert result["deterministic_replay"] is True
    assert result["diagnostics"]["changed_state_key_count"] > 0
    encoder_before = {k: v for k, v in initial_state.items() if not k.startswith("fc.")}
    encoder_after = {k: v for k, v in result["final_state"].items() if not k.startswith("fc.")}
    changed_encoder_keys = [k for k in encoder_before if not torch.equal(encoder_before[k], encoder_after[k])]
    assert changed_encoder_keys, "encoder (including BatchNorm buffers) should legitimately change in the full-network arm"


def test_warm_full_network_rejects_non_deterministic_batch_factory():
    torch.manual_seed(0)
    initial_state = _model_factory().state_dict()
    calls = {"n": 0}

    def flaky_factory(step: int, seed: int):
        calls["n"] += 1
        generator = torch.Generator().manual_seed(seed * 100000 + step + calls["n"])  # not pure
        inputs = torch.randn(8, 5, generator=generator)
        labels = torch.randint(0, 2, (8,), generator=generator)
        return inputs, labels

    with pytest.raises(RuntimeError, match="deterministic replay"):
        warm_full_network(_model_factory, initial_state, flaky_factory, _forward_logits, seed=11)


def test_head_perturbation_control_norm_matches_a_real_warmup_displacement():
    torch.manual_seed(0)
    before = _model_factory()
    before_weight, before_bias = before.fc.weight.detach().clone(), before.fc.bias.detach().clone()
    # Simulate a "real warm-up" head displacement directly (cheaper than a
    # full descriptor/coordinate warm-up run for this test's purpose: only
    # the norm-matching contract with apply_head_perturbation_control is
    # under test here).
    with torch.no_grad():
        before.fc.weight.add_(0.1)
        before.fc.bias.add_(0.05)
    real_warmup_norm = head_displacement_norm(before_weight, before_bias, before.fc.weight, before.fc.bias)

    from alignn_stage2.activation_probe import apply_head_perturbation_control
    control_model = _model_factory()
    control_model.load_state_dict(_model_factory().state_dict())
    control_before_weight = control_model.fc.weight.detach().clone()
    control_before_bias = control_model.fc.bias.detach().clone()
    apply_head_perturbation_control(control_model, target_norm=real_warmup_norm, seed=99)
    achieved = head_displacement_norm(control_before_weight, control_before_bias,
        control_model.fc.weight.detach(), control_model.fc.bias.detach())
    assert math.isclose(achieved, real_warmup_norm, rel_tol=1e-4, abs_tol=1e-6)
