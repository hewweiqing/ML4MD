"""The new SCRATCH-condition whole-network Random2 warm-up primitive
(Random2-Full).

Structurally different from the existing head-only warm-ups: those never
re-invoke the encoder after one .eval()+no_grad() descriptor-extraction
pass, so BatchNorm running stats can't change (see research finding in the
plan). Here every parameter is unfrozen and the whole model is run in
.train() mode on noise each step, so BatchNorm running stats legitimately
update. That existing "BatchNorm buffers must stay byte-identical" gate
would be simply wrong here — it is replaced, not reused, with the
invariants below.

Hyperparameters are fixed by a stated principle, not tuned to produce an
effect: identical optimizer-step/batch-size budget to the existing
head-only arms (WARMUP_STEPS=938, BATCH_SIZE=128), only the learning rate
differs in magnitude reasoning (LEARNING_RATE=1e-4, same as the head-only
arms too) because the only thing that should differ between arms is what's
trainable, not how much compute each gets. See FULL_NETWORK_WARMUP_CONFIG.json.
938 whole-network steps is a much larger intervention than 938 steps on a
256->2 head alone and may substantially degrade the encoder; if that
happens it must be reported as a finding, not used as a reason to shrink
this budget after the fact.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import torch

WARMUP_STEPS = 938
BATCH_SIZE = 128
LEARNING_RATE = 1e-4


def _all_finite(model: torch.nn.Module) -> bool:
    return all(torch.isfinite(value).all() for value in model.state_dict().values()
        if torch.is_floating_point(value))


def _warm_once(model_factory: Callable[[], torch.nn.Module], initial_state: dict[str, torch.Tensor],
        graph_batch_factory: Callable[[int, int], tuple[Any, torch.Tensor]],
        forward_logits_fn: Callable[[torch.nn.Module, Any], torch.Tensor], *, seed: int) -> tuple[dict, dict]:
    model = model_factory()
    model.load_state_dict(initial_state)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    for step in range(WARMUP_STEPS):
        batch_input, labels = graph_batch_factory(step, seed)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = forward_logits_fn(model, batch_input)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        if not _all_finite(model):
            raise RuntimeError(f"non-finite parameter/buffer after full-network warm-up step {step}")
    final_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    changed = {name for name in initial_state if not torch.equal(initial_state[name].cpu(), final_state[name].cpu())}
    if not changed:
        raise RuntimeError("full-network warm-up did not change any parameter or buffer — encoder should have changed")
    return final_state, {"changed_state_key_count": len(changed), "step_count": WARMUP_STEPS}


def warm_full_network(model_factory: Callable[[], torch.nn.Module], initial_state: dict[str, torch.Tensor],
        graph_batch_factory: Callable[[int, int], tuple[Any, torch.Tensor]],
        forward_logits_fn: Callable[[torch.nn.Module, Any], torch.Tensor], *, seed: int) -> dict[str, Any]:
    """graph_batch_factory(step, seed) -> (batch_input, labels): must be a
    pure function of (step, seed) so the deterministic-replay check below is
    meaningful — each call should freshly resample noise features AND
    labels (per Stage A's instruction: resampled each iteration, inputs and
    labels unpaired), but reproducibly given the same (step, seed).
    """
    initial_state = {name: value.detach().clone() for name, value in initial_state.items()}
    first_state, first_diag = _warm_once(model_factory, initial_state, graph_batch_factory, forward_logits_fn, seed=seed)
    second_state, second_diag = _warm_once(model_factory, initial_state, graph_batch_factory, forward_logits_fn, seed=seed)
    identical = all(torch.equal(first_state[key].cpu(), second_state[key].cpu()) for key in first_state)
    if not identical or first_diag != second_diag:
        raise RuntimeError("full-network warm-up failed deterministic replay")
    return {"final_state": first_state, "diagnostics": first_diag, "deterministic_replay": True,
        "seed": seed, "optimizer_steps": WARMUP_STEPS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "initial_state_sha256_note": "caller records pre/post full model-state SHA-256 for provenance"}
