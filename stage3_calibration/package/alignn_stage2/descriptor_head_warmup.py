"""Head-only, descriptor-style Random2 warm-up primitive for Stage A.

Clean reimplementation of the mechanism in descriptor_v41's training.py /
coordinate_gpu_v4's dead-code create_start_states() (real-structure
descriptors, synthetic Gaussian resample around their mean/std, balanced
random labels, AdamW on fc only) — not copied forward as-is, and using this
package's own seed-offset namespace so its RNG streams never collide with
either existing package's.

Two layers, matching the existing convention split between "extract
descriptors from the real model" (needs a real ALIGNN forward pass, cannot
be exercised locally) and "warm the head given descriptors" (pure torch,
locally testable against a synthetic model).
"""
from __future__ import annotations

from copy import deepcopy

import torch

DESCRIPTOR_WARMUP_SEED_STREAM = "stage3-descriptor-warmup"
DESCRIPTOR_HOLDOUT_SEED_STREAM = "stage3-descriptor-holdout"
SYNTHETIC_COUNT = 3000
WARMUP_STEPS = 938
BATCH_SIZE = 128
LEARNING_RATE = 1e-4


def extract_pooled_descriptors(model: torch.nn.Module, atom_graphs: list, line_graphs: list, batch_fn) -> torch.Tensor:
    """batch_fn(atom_graphs, line_graphs, indices) -> model input batch, same
    contract as gpu_coordinate_training.py's graph_batch(). Runs the full
    model in eval/no_grad, captures fc's input via a forward hook (the
    pooled per-structure descriptor), and asserts the encoder/buffers are
    unchanged by this one-time extraction pass.
    """
    model.eval()
    if model.training:
        raise RuntimeError("descriptor extraction must run with the full model in eval mode")
    encoder_before = {name: value.detach().clone() for name, value in model.state_dict().items()
        if not name.startswith("fc.")}
    descriptors = []
    with torch.no_grad():
        for start in range(0, len(atom_graphs), 32):
            indices = list(range(start, min(start + 32, len(atom_graphs))))
            captured = []
            hook = model.fc.register_forward_hook(lambda _module, inputs, _output: captured.append(inputs[0]))
            model(batch_fn(atom_graphs, line_graphs, indices))
            hook.remove()
            batch = captured[-1]
            if batch.ndim == 1:
                batch = batch.reshape(1, -1)
            descriptors.append(batch.detach())
    descriptors = torch.cat(descriptors)
    encoder_after = {name: value.detach() for name, value in model.state_dict().items() if not name.startswith("fc.")}
    for name, before in encoder_before.items():
        if not torch.equal(before, encoder_after[name]):
            raise RuntimeError(f"descriptor extraction modified non-head state: {name}")
    if not torch.isfinite(descriptors).all():
        raise RuntimeError("non-finite descriptors extracted")
    return descriptors


def _warm_once(fc_state: dict, descriptors: torch.Tensor, *, seed: int) -> tuple[dict, dict]:
    fc = torch.nn.Linear(descriptors.shape[1], 2)
    fc.load_state_dict(fc_state)
    mean, std = descriptors.mean(0), descriptors.std(0, unbiased=False).clamp_min(1e-6)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    holdout_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    synthetic = mean + std * torch.randn((SYNTHETIC_COUNT, descriptors.shape[1]), generator=generator)
    labels = torch.tensor([0, 1] * (SYNTHETIC_COUNT // 2))
    permutation = torch.randperm(SYNTHETIC_COUNT, generator=generator)
    synthetic, labels = synthetic[permutation], labels[permutation]
    optimizer = torch.optim.AdamW(fc.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    cursor = 0
    for _ in range(WARMUP_STEPS):
        if cursor + BATCH_SIZE > SYNTHETIC_COUNT:
            permutation = torch.randperm(SYNTHETIC_COUNT, generator=generator)
            cursor = 0
        indices, cursor = permutation[cursor:cursor + BATCH_SIZE], cursor + BATCH_SIZE
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(fc(synthetic[indices]), labels[indices]).backward()
        optimizer.step()
    heldout = mean + std * torch.randn((SYNTHETIC_COUNT, descriptors.shape[1]), generator=holdout_generator)
    with torch.no_grad():
        probabilities = torch.softmax(fc(heldout), dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-15).log()).sum(1)
    diagnostics = {"mean_positive_probability": float(probabilities[:, 1].mean()),
        "mean_entropy": float(entropy.mean()), "finite": bool(torch.isfinite(probabilities).all())}
    return fc.state_dict(), diagnostics


def warm_descriptor_head(fc_state: dict, descriptors: torch.Tensor, *, seed: int) -> dict:
    """Runs the warm-up twice from the same seed (deterministic-replay
    contract) and returns the final fc state plus diagnostics, hard-failing
    if the two runs disagree.
    """
    first_state, first_diag = _warm_once(fc_state, descriptors, seed=seed)
    second_state, second_diag = _warm_once(fc_state, descriptors, seed=seed)
    identical = all(torch.equal(first_state[key], second_state[key]) for key in first_state)
    if not identical or first_diag != second_diag:
        raise RuntimeError("descriptor head warm-up failed deterministic replay")
    return {"fc_state": first_state, "diagnostics": first_diag, "deterministic_replay": True,
        "seed": seed, "optimizer_steps": WARMUP_STEPS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "seed_streams": [DESCRIPTOR_WARMUP_SEED_STREAM, DESCRIPTOR_HOLDOUT_SEED_STREAM]}
