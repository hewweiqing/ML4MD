"""Head-only, coordinate-style Random2 warm-up primitive for Stage A.

Clean reimplementation of coordinate_gpu_v4's create_coordinate_start_states
mechanism (train fc only, on descriptors extracted from real
coordinate-perturbed structures with balanced random labels), scoped as a
reusable primitive and using this package's own seed-offset namespace
(derive_stream_seed with STAGE3_VERSION, not coordinate_gpu_v4's `+280000`
magic offset) so RNG streams never collide across packages.

Reuses extract_pooled_descriptors from descriptor_head_warmup.py (an
intra-package import — matches the repo convention of cross-referencing
within one package while never importing across package boundaries).
"""
from __future__ import annotations

import torch

from .descriptor_head_warmup import extract_pooled_descriptors
from .random_feature_graphs import derive_stream_seed

WARMUP_STEPS = 938
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
RECORD_COUNT = 3000


def _warm_once(fc_state: dict, descriptors: torch.Tensor, labels: torch.Tensor, *, seed: int) -> dict:
    device = descriptors.device
    labels = labels.to(device)
    fc = torch.nn.Linear(descriptors.shape[1], 2).to(device)
    fc.load_state_dict({key: value.to(device) for key, value in fc_state.items()})
    optimizer = torch.optim.AdamW(fc.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    # Permutation RNG stays pinned to CPU (matches coordinate_gpu_v4's
    # gpu_coordinate_training.py convention: "Only the selected batch
    # indices cross to CUDA; the permutation RNG does not").
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(RECORD_COUNT, generator=generator).to(device)
    cursor = 0
    for _ in range(WARMUP_STEPS):
        if cursor + BATCH_SIZE > RECORD_COUNT:
            permutation = torch.randperm(RECORD_COUNT, generator=generator).to(device)
            cursor = 0
        indices, cursor = permutation[cursor:cursor + BATCH_SIZE], cursor + BATCH_SIZE
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(fc(descriptors[indices]), labels[indices]).backward()
        optimizer.step()
    with torch.no_grad():
        probabilities = torch.softmax(fc(descriptors), dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-15).log()).sum(1)
    return fc.state_dict(), {"mean_positive_probability": float(probabilities[:, 1].mean()),
        "mean_entropy": float(entropy.mean()), "finite": bool(torch.isfinite(probabilities).all())}


def warm_coordinate_head(fc_state: dict, descriptors: torch.Tensor, labels: torch.Tensor, *,
        version: int, fold: int, seed: int) -> dict:
    """descriptors/labels: exactly RECORD_COUNT rows, extracted from
    coordinate-perturbed (Stage A input set (b)) structures — labels are
    `balanced_random_labels` output, never true labels. Runs the warm-up
    twice from the same derived seed (deterministic-replay contract),
    hard-failing if the two runs disagree.
    """
    if descriptors.shape[0] != RECORD_COUNT or labels.shape[0] != RECORD_COUNT:
        raise RuntimeError(f"coordinate head warm-up requires exactly {RECORD_COUNT} records")
    if sum(int(x) for x in labels) != RECORD_COUNT // 2:
        raise RuntimeError("coordinate head warm-up labels are not exactly balanced")
    permutation_seed = derive_stream_seed(version, fold, seed, 0, "stage3-coordinate-warmup")
    first_state, first_diag = _warm_once(fc_state, descriptors, labels, seed=permutation_seed)
    second_state, second_diag = _warm_once(fc_state, descriptors, labels, seed=permutation_seed)
    identical = all(torch.equal(first_state[key], second_state[key]) for key in first_state)
    if not identical or first_diag != second_diag:
        raise RuntimeError("coordinate head warm-up failed deterministic replay")
    return {"fc_state": first_state, "diagnostics": first_diag, "deterministic_replay": True,
        "permutation_seed": permutation_seed, "optimizer_steps": WARMUP_STEPS,
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "record_count": RECORD_COUNT}


__all__ = ["extract_pooled_descriptors", "warm_coordinate_head", "WARMUP_STEPS", "BATCH_SIZE", "LEARNING_RATE", "RECORD_COUNT"]
