import inspect

import torch

from alignn_stage2 import activation_probe
from alignn_stage2.activation_probe import logit_diagnostics, mechanism_verdict


def test_verdict_overconfident_case():
    logits = torch.tensor([[-5.0, 5.0]] * 100)
    diagnostics = logit_diagnostics(logits)
    assert mechanism_verdict(diagnostics) == "overconfidence_present"


def test_verdict_near_chance_case():
    torch.manual_seed(0)
    logits = torch.randn(500, 2) * 0.02
    diagnostics = logit_diagnostics(logits)
    assert mechanism_verdict(diagnostics) == "no_initial_overconfidence"


def test_verdict_ambiguous_case():
    logits = torch.tensor([[0.0, 1.2]] * 100)
    diagnostics = logit_diagnostics(logits)
    assert mechanism_verdict(diagnostics) == "ambiguous_see_full_distribution"


def _body_only(function) -> str:
    """Source with the docstring stripped, so a prose mention of the other
    function's name (for documentation purposes) doesn't trip a call-site
    check the way an actual reference/call would.
    """
    lines = inspect.getsource(function).splitlines()
    doc = inspect.getdoc(function)
    if doc:
        # Drop every physical source line whose stripped text is part of the
        # docstring, robust to the docstring being on one or many lines.
        doc_lines = {line.strip() for line in doc.splitlines() if line.strip()}
        lines = [line for line in lines if line.strip() not in doc_lines
            and line.strip() not in {'"""', "'''"}]
    return "\n".join(lines)


def test_mechanism_verdict_source_never_calls_the_mechanical_gate():
    """Mechanical-gate-neutrality pattern (matches descriptor_v41's
    test_v17_scientific_integrity.py::test_validation_adequacy_gate_is_outcome_neutral_to_random2):
    the verdict computation must never read the separate pipeline-precondition
    gate, checked by asserting the gate is never called from the verdict
    function's own source body (docstring mentions for documentation don't
    count as a violation).
    """
    body = _body_only(mechanism_verdict)
    assert "mechanical_gate(" not in body
    assert "evaluate_mechanical_gate(" not in body


def test_evaluate_mechanical_gate_source_never_calls_the_verdict():
    body = _body_only(activation_probe.evaluate_mechanical_gate)
    assert "mechanism_verdict(" not in body
