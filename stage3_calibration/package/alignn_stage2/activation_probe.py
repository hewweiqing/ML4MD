"""Stage A measurement primitives: logit/confidence diagnostics, an
execution-order-safe per-leaf-module activation-RMS logger, the decision
criterion, and the head-perturbation control.

ALIGNN's exact internal module names/nesting are not verified in this
environment (see STAGE3_STAGE_A_PROTOCOL.md). The activation-RMS logger is
deliberately name-agnostic: it hooks every leaf module found via
`model.named_modules()` and records call order using a counter incremented
inside the hook itself at fire time, not via `named_modules()`'s own
enumeration order. PyTorch fires forward hooks in true execution order
regardless of module-tree traversal order, so this ordering guarantee does
not depend on knowing ALIGNN's internal names in advance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


def leaf_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    return [(name, module) for name, module in model.named_modules() if not list(module.children())]


@dataclass
class ActivationTrace:
    records: list[dict[str, Any]] = field(default_factory=list)
    _counter: list[int] = field(default_factory=lambda: [0])

    def hook_factory(self, name: str):
        def hook(module: torch.nn.Module, _inputs, output) -> None:
            call_order_index = self._counter[0]
            self._counter[0] += 1
            tensors = _flatten_tensors(output)
            if not tensors:
                self.records.append({"call_order_index": call_order_index, "module_name": name,
                    "module_type": type(module).__name__, "output_rms": None,
                    "is_batchnorm": "BatchNorm" in type(module).__name__, "note": "non-tensor output"})
                return
            values = torch.cat([tensor.detach().float().reshape(-1) for tensor in tensors])
            rms = float(torch.sqrt(torch.mean(values ** 2)).item()) if values.numel() else None
            self.records.append({"call_order_index": call_order_index, "module_name": name,
                "module_type": type(module).__name__, "output_rms": rms,
                "is_batchnorm": "BatchNorm" in type(module).__name__})
        return hook


def _flatten_tensors(output: Any) -> list[torch.Tensor]:
    if torch.is_tensor(output):
        return [output]
    if isinstance(output, (list, tuple)):
        flattened: list[torch.Tensor] = []
        for item in output:
            flattened.extend(_flatten_tensors(item))
        return flattened
    if isinstance(output, dict):
        flattened = []
        for item in output.values():
            flattened.extend(_flatten_tensors(item))
        return flattened
    return []


def register_activation_probe(model: torch.nn.Module) -> tuple[ActivationTrace, list]:
    """Attach a forward hook to every leaf module. Caller must remove the
    returned handles after the forward pass (they are not auto-removed, so
    repeated forward passes with a fresh ActivationTrace don't double-fire).
    """
    trace = ActivationTrace()
    handles = [module.register_forward_hook(trace.hook_factory(name)) for name, module in leaf_modules(model)]
    return trace, handles


def remove_hooks(handles: list) -> None:
    for handle in handles:
        handle.remove()


def activation_scale_summary(trace: ActivationTrace) -> dict[str, Any]:
    ordered = sorted((row for row in trace.records if row["output_rms"] is not None),
        key=lambda row: row["call_order_index"])
    if not ordered:
        return {"leaf_module_count": len(trace.records), "measured_count": 0,
            "monotonic_non_increasing": None, "batchnorm_modules_present": False}
    rms_sequence = [row["output_rms"] for row in ordered]
    batchnorm_present = any(row["is_batchnorm"] for row in trace.records)
    non_increasing = all(later <= earlier + 1e-6 for earlier, later in zip(rms_sequence, rms_sequence[1:]))
    return {"leaf_module_count": len(trace.records), "measured_count": len(ordered),
        "rms_by_call_order": ordered, "rms_first": rms_sequence[0], "rms_last": rms_sequence[-1],
        "rms_max": max(rms_sequence), "rms_min": min(rms_sequence),
        "monotonic_non_increasing": non_increasing, "batchnorm_modules_present": batchnorm_present,
        "gradual_reduction_structurally_possible_note":
            "ALIGNN uses BatchNorm throughout; BatchNorm renormalizes scale at every "
            "occurrence, so a strictly monotonic 'gradual reduction of activation scale "
            "across the hierarchy' (as the paper describes for its unnormalized backbone) "
            "is not structurally expected here regardless of what this trace shows for one pass."}


def logit_diagnostics(logits: torch.Tensor) -> dict[str, Any]:
    """logits: (N, 2) native pre-softmax logits (class-0, class-1 order)."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise RuntimeError("logit_diagnostics expects native two-class logits of shape (N, 2)")
    logits = logits.detach().float()
    gap = (logits[:, 1] - logits[:, 0]).abs()
    probabilities = torch.softmax(logits, dim=1)
    max_probability = probabilities.max(dim=1).values
    predicted_label = (logits[:, 1] > logits[:, 0]).long()
    entropy = -(probabilities * probabilities.clamp_min(1e-15).log()).sum(dim=1)
    histogram_edges = np.linspace(0.5, 1.0, 11)
    histogram_counts, _ = np.histogram(max_probability.detach().cpu().numpy(), bins=histogram_edges)
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite logits in logit_diagnostics")
    return {
        "n": int(logits.shape[0]),
        "logit_mean": [float(logits[:, 0].mean()), float(logits[:, 1].mean())],
        "logit_std": [float(logits[:, 0].std(unbiased=False)), float(logits[:, 1].std(unbiased=False))],
        "logit_min": [float(logits[:, 0].min()), float(logits[:, 1].min())],
        "logit_max": [float(logits[:, 0].max()), float(logits[:, 1].max())],
        "logit_gap_mean": float(gap.mean()), "logit_gap_std": float(gap.std(unbiased=False)),
        "max_softmax_mean": float(max_probability.mean()),
        "max_softmax_fraction_above_0.9": float((max_probability > 0.9).float().mean()),
        "max_softmax_histogram": {"bin_edges": histogram_edges.tolist(), "counts": histogram_counts.tolist()},
        "predictive_entropy_mean": float(entropy.mean()), "predictive_entropy_std": float(entropy.std(unbiased=False)),
        "predicted_class_1_fraction": float(predicted_label.float().mean()),
    }


# Decision-criterion thresholds, stated explicitly so the classification is
# auditable, not implicit. Chosen to match the qualitative description in
# STAGE3_STAGE_A_PROTOCOL.md ("substantially above 0.5 with mass piling
# toward 1.0" vs. "near 0.5 with small logit gaps"), not tuned against any
# measured outcome (no Stage A output exists yet in this environment).
OVERCONFIDENCE_MAX_SOFTMAX_MEAN_THRESHOLD = 0.60
OVERCONFIDENCE_FRACTION_ABOVE_0_9_THRESHOLD = 0.05
NO_OVERCONFIDENCE_MAX_SOFTMAX_MEAN_THRESHOLD = 0.55
NO_OVERCONFIDENCE_LOGIT_GAP_THRESHOLD = 1.0


def mechanism_verdict(diagnostics: dict[str, Any]) -> str:
    """Returns one of: 'overconfidence_present', 'no_initial_overconfidence',
    'ambiguous_see_full_distribution'. Deliberately a three-way call: forcing
    a binary answer in the middle of the threshold gap would misrepresent an
    honestly ambiguous measurement as a clean finding.
    """
    mean_max_softmax = diagnostics["max_softmax_mean"]
    fraction_high = diagnostics["max_softmax_fraction_above_0.9"]
    logit_gap = diagnostics["logit_gap_mean"]
    if mean_max_softmax >= OVERCONFIDENCE_MAX_SOFTMAX_MEAN_THRESHOLD and fraction_high >= OVERCONFIDENCE_FRACTION_ABOVE_0_9_THRESHOLD:
        return "overconfidence_present"
    if mean_max_softmax <= NO_OVERCONFIDENCE_MAX_SOFTMAX_MEAN_THRESHOLD and logit_gap <= NO_OVERCONFIDENCE_LOGIT_GAP_THRESHOLD:
        return "no_initial_overconfidence"
    return "ambiguous_see_full_distribution"


def evaluate_mechanical_gate(mean_positive_probability: float, mean_entropy: float) -> dict[str, Any]:
    """The existing [0.45,0.55]/entropy>=0.68 post-warm-up precondition gate,
    kept structurally separate from mechanism_verdict — never read by it, and
    never cited elsewhere as evidence for the paper's mechanism.
    """
    passed = (0.45 <= mean_positive_probability <= 0.55) and mean_entropy >= 0.68 and mean_entropy <= math.log(2) + 1e-6
    return {"mean_positive_probability": mean_positive_probability, "mean_entropy": mean_entropy,
        "thresholds": {"mean_positive_probability_range": [0.45, 0.55], "mean_entropy_min": 0.68},
        "status": "passed" if passed else "failed",
        "note": "pipeline precondition only (proves the warm-up did something non-degenerate); "
                "never used as evidence for the paper's mechanism"}


def head_displacement_norm(before_weight: torch.Tensor, before_bias: torch.Tensor,
        after_weight: torch.Tensor, after_bias: torch.Tensor) -> float:
    delta = torch.cat([(after_weight - before_weight).reshape(-1), (after_bias - before_bias).reshape(-1)])
    return float(torch.linalg.vector_norm(delta.float()))


def apply_head_perturbation_control(model: torch.nn.Module, *, target_norm: float, seed: int) -> dict[str, Any]:
    """No forward/backward pass, no noise data at all: displaces fc.weight
    and fc.bias by an independent random direction scaled to `target_norm`
    (matched to a real warm-up arm's observed displacement norm for the
    same seed). This is the control that distinguishes "the warm-up
    mechanism specifically did something" from "any head displacement of
    this size does the same thing."
    """
    fc = model.fc
    before_weight, before_bias = fc.weight.detach().clone(), fc.bias.detach().clone()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    direction = torch.cat([torch.randn(before_weight.numel(), generator=generator),
        torch.randn(before_bias.numel(), generator=generator)])
    direction = direction / torch.linalg.vector_norm(direction)
    displacement = direction * float(target_norm)
    weight_delta = displacement[:before_weight.numel()].reshape(before_weight.shape).to(before_weight.dtype)
    bias_delta = displacement[before_weight.numel():].reshape(before_bias.shape).to(before_bias.dtype)
    with torch.no_grad():
        fc.weight.add_(weight_delta)
        fc.bias.add_(bias_delta)
    achieved_norm = head_displacement_norm(before_weight, before_bias, fc.weight.detach(), fc.bias.detach())
    if not math.isclose(achieved_norm, float(target_norm), rel_tol=1e-4, abs_tol=1e-6):
        raise RuntimeError(f"head-perturbation control norm mismatch: target={target_norm}, achieved={achieved_norm}")
    return {"condition": "head_perturbation_control", "seed": seed, "target_norm": float(target_norm),
        "achieved_norm": achieved_norm, "no_noise_data_used": True, "no_forward_backward_pass": True}
