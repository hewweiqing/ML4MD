"""Final MUBen-derived ALIGNN scalar-temperature extension.

MUBen's completed exact-reproduction audit is the evidence source; ALIGNN's
prospectively frozen extension requires the safer T=exp(log_T) parameterization.
This adapter fits one shared scalar to native [n,2] logits in float64 and never
accepts test data or probabilities.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

IMPLEMENTATION_VERSION = "muben-final-446471d-alignn-logt-float64-lbfgs-v2-grad1e-7"
MAX_ITERATIONS = 500
TOLERANCE_GRAD = 1e-7
TOLERANCE_CHANGE = 1e-12


def _inputs(validation_logits, validation_labels):
    logits = np.asarray(validation_logits)
    labels = np.asarray(validation_labels)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("validation_logits must have shape [n, 2]")
    if labels.ndim != 1 or len(labels) != len(logits):
        raise ValueError("validation labels must be aligned to logits")
    if not np.issubdtype(logits.dtype, np.floating) or not np.isfinite(logits).all():
        raise ValueError("validation logits must be finite floating-point values")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("validation labels must be binary")
    return logits, labels.astype(np.int64, copy=False)


def _loss(logits, labels, log_temperature):
    return F.cross_entropy(logits / torch.exp(log_temperature), labels, reduction="mean")


def fit_temperature(validation_logits, validation_labels):
    """Minimize validation NLL over one positive T=exp(log_T) scalar."""
    logits, labels = _inputs(validation_logits, validation_labels)
    torch.manual_seed(0)
    logits_t = torch.as_tensor(logits, dtype=torch.float64, device="cpu")
    labels_t = torch.as_tensor(labels, dtype=torch.long, device="cpu")
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=1.0, max_iter=MAX_ITERATIONS,
        tolerance_grad=TOLERANCE_GRAD, tolerance_change=TOLERANCE_CHANGE,
        history_size=50, line_search_fn="strong_wolfe")
    with torch.no_grad():
        nll_before = float(_loss(logits_t, labels_t, log_temperature).item())
    loss_curve = []

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(logits_t, labels_t, log_temperature)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite validation NLL during temperature fitting")
        loss.backward()
        loss_curve.append(float(loss.detach().item()))
        return loss

    optimizer.step(closure)
    optimizer.zero_grad(set_to_none=True)
    final_loss = _loss(logits_t, labels_t, log_temperature)
    final_loss.backward()
    gradient = float(abs(log_temperature.grad.detach().item()))
    with torch.no_grad():
        temperature = float(torch.exp(log_temperature).item())
        nll_after = float(final_loss.item())
    finite = all(math.isfinite(value) for value in (temperature, nll_before, nll_after, gradient))
    converged = bool(finite and temperature > 0 and gradient <= TOLERANCE_GRAD
        and nll_after <= nll_before + TOLERANCE_CHANGE)
    return {"temperature": temperature,
        "objective": "unweighted_inner_validation_nll_two_class_cross_entropy",
        "converged": converged, "optimization_steps": len(loss_curve),
        "validation_nll_before": nll_before, "validation_nll_after": nll_after,
        "implementation_version": IMPLEMENTATION_VERSION, "initial_temperature": 1.0,
        "parameterization": "T=exp(log_T)", "numerical_dtype": "torch.float64",
        "optimizer": "LBFGS_strong_wolfe", "max_iterations": MAX_ITERATIONS,
        "tolerance_grad": TOLERANCE_GRAD, "tolerance_change": TOLERANCE_CHANGE,
        "final_abs_log_T_gradient": gradient, "loss_curve": loss_curve,
        "fit_split": "inner_validation_only", "convergence_status": "converged" if converged else "not_converged"}


def apply_temperature(logits, fitted_object):
    array = np.asarray(logits)
    temperature = float(fitted_object["temperature"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    return array / temperature
