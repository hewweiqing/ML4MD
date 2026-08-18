import math

import torch

from alignn_stage2.activation_probe import (activation_scale_summary, apply_head_perturbation_control,
    evaluate_mechanical_gate, head_displacement_norm, logit_diagnostics, mechanism_verdict,
    register_activation_probe, remove_hooks)


class _OutOfOrderModel(torch.nn.Module):
    """Declaration order (`second` before `first`) deliberately differs from
    forward-execution order, so a test that only trusted named_modules()
    enumeration order would get this wrong.
    """
    def __init__(self) -> None:
        super().__init__()
        self.second = torch.nn.Linear(3, 3)
        self.first = torch.nn.Linear(3, 3)
        self.norm = torch.nn.BatchNorm1d(3)
        self.fc = torch.nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.first(x)
        b = self.norm(a)
        c = self.second(b)
        return self.fc(c)


def test_hooks_fire_in_true_execution_order_not_declaration_order():
    torch.manual_seed(0)
    model = _OutOfOrderModel().eval()
    trace, handles = register_activation_probe(model)
    with torch.no_grad():
        model(torch.randn(5, 3))
    remove_hooks(handles)
    ordered = sorted(trace.records, key=lambda row: row["call_order_index"])
    names = [row["module_name"] for row in ordered]
    assert names == ["first", "norm", "second", "fc"]


def test_activation_scale_summary_flags_batchnorm_and_reports_rms():
    torch.manual_seed(1)
    model = _OutOfOrderModel().eval()
    trace, handles = register_activation_probe(model)
    with torch.no_grad():
        model(torch.randn(8, 3))
    remove_hooks(handles)
    summary = activation_scale_summary(trace)
    assert summary["batchnorm_modules_present"] is True
    assert summary["measured_count"] == 4
    assert summary["rms_max"] >= summary["rms_min"] >= 0


def test_logit_diagnostics_overconfident_case():
    logits = torch.tensor([[-6.0, 6.0]] * 50 + [[6.0, -6.0]] * 50)
    diagnostics = logit_diagnostics(logits)
    assert diagnostics["max_softmax_mean"] > 0.99
    assert mechanism_verdict(diagnostics) == "overconfidence_present"


def test_logit_diagnostics_near_chance_case():
    torch.manual_seed(2)
    logits = torch.randn(200, 2) * 0.05
    diagnostics = logit_diagnostics(logits)
    verdict = mechanism_verdict(diagnostics)
    assert verdict in {"no_initial_overconfidence", "ambiguous_see_full_distribution"}
    assert diagnostics["max_softmax_mean"] < 0.6


def test_mechanism_verdict_ambiguous_case_is_not_forced_binary():
    logits = torch.tensor([[0.0, 1.4]] * 100)
    diagnostics = logit_diagnostics(logits)
    assert mechanism_verdict(diagnostics) == "ambiguous_see_full_distribution"


def test_mechanical_gate_pass_and_fail():
    passed = evaluate_mechanical_gate(0.5, 0.69)
    assert passed["status"] == "passed"
    failed = evaluate_mechanical_gate(0.9, 0.69)
    assert failed["status"] == "failed"


def test_head_perturbation_control_matches_target_norm():
    torch.manual_seed(3)
    model = _OutOfOrderModel()
    before_weight, before_bias = model.fc.weight.detach().clone(), model.fc.bias.detach().clone()
    provenance = apply_head_perturbation_control(model, target_norm=0.37, seed=42)
    achieved = head_displacement_norm(before_weight, before_bias, model.fc.weight.detach(), model.fc.bias.detach())
    assert math.isclose(achieved, 0.37, rel_tol=1e-4, abs_tol=1e-6)
    assert provenance["no_noise_data_used"] is True
    assert provenance["no_forward_backward_pass"] is True


def test_head_perturbation_control_is_deterministic_given_same_seed():
    torch.manual_seed(4)
    model_a, model_b = _OutOfOrderModel(), _OutOfOrderModel()
    model_b.load_state_dict(model_a.state_dict())
    apply_head_perturbation_control(model_a, target_norm=0.5, seed=7)
    apply_head_perturbation_control(model_b, target_norm=0.5, seed=7)
    assert torch.equal(model_a.fc.weight, model_b.fc.weight)
    assert torch.equal(model_a.fc.bias, model_b.fc.bias)
