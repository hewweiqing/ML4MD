"""Cache-once temperature-scaling fitting for the Phase 3 exact-reproduction track.

Unlike Phase 2's DNN-RDKit architecture (which called MUBen's native `ts_session()` -- a
second, live forward pass over the validation set each time), Phase 3 caches raw logits
exactly once per (dataset, seed, backbone) via `Trainer.inference()` (itself fully native --
see trainer.py's `inference()` method), then fits `T` and evaluates BOTH raw and calibrated
metrics from that single cached array. Positive-scalar ranking invariance is then true by
mathematical construction, not just empirically verified.

This module still uses MUBen's own `TSModel` class and the exact masked-BCE loss formula
`Trainer.get_loss()` uses (reproduced here because `get_loss()` requires a live `Batch` object
with molecular inputs, which cached logits don't have -- feeding a stand-in "model" that
indexes into the cached array in place of a real forward pass is the same technique already
validated bit-exact against native `TSModel`+`AdamW` training in
`tests/test_calibration_matches_native.py`). No exact-reproduction-track calibration.py
extension-track code is imported or used here.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

import _pathsetup  # noqa: F401
from muben.uncertainty.ts import TSModel


class _CachedLogitsReplay(torch.nn.Module):
    """Stand-in 'model' for TSModel: given a batch of integer row indices, returns the
    corresponding rows of a precomputed logits tensor -- no real forward pass.
    """

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("_logits", logits)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self._logits[idx]


@dataclass
class CachedTemperatureFitResult:
    temperature: np.ndarray
    loss_curve: list
    n_epochs: int
    lr: float
    batch_size: int
    optimizer_betas: tuple
    optimizer_eps: float


def fit_temperature_from_cached_logits(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    val_masks: np.ndarray,
    n_tasks: int,
    n_lbs: int = 1,
    lr: float = 0.01,
    n_epochs: int = 20,
    batch_size: Optional[int] = None,
    optimizer_betas: tuple = (0.9, 0.999),
    optimizer_eps: float = 1e-8,
    seed: int = 0,
    device: str = "cpu",
) -> CachedTemperatureFitResult:
    """Fits MUBen's native per-task TSModel temperature using CACHED validation logits --
    exactly one forward pass' worth of data, reused across every optimization step, instead of
    a second live forward pass per epoch/batch as native ts_session() would do.

    `optimizer_betas`/`optimizer_eps` must match the Trainer subclass's own
    `initialize_optimizer()` override (e.g. TrainerUnimol uses betas=(0.9, 0.99), eps=1e-6,
    not AdamW's plain defaults) for a faithful reproduction of that backbone's actual TS
    optimization dynamics.

    No test data can reach this function -- only `val_*` arrays are accepted.
    """
    if not (val_logits.shape == val_labels.shape == val_masks.shape):
        raise ValueError("val_logits, val_labels, val_masks must share shape (N, n_tasks)")

    torch.manual_seed(seed)

    logits_t = torch.as_tensor(val_logits, dtype=torch.float32, device=device)
    labels_t = torch.as_tensor(val_labels, dtype=torch.float32, device=device)
    masks_t = torch.as_tensor(val_masks, dtype=torch.float32, device=device)

    replay_model = _CachedLogitsReplay(logits_t)
    ts_model = TSModel(replay_model, n_tasks)

    optimizer = AdamW([ts_model.temperature], lr=lr, betas=optimizer_betas, eps=optimizer_eps)

    n = logits_t.shape[0]
    bs = batch_size or n
    idx_all = torch.arange(n, device=device)
    loss_curve = []

    for _epoch in range(n_epochs):
        perm = idx_all if bs >= n else idx_all[torch.randperm(n, device=device)]
        epoch_loss = 0.0
        epoch_count = 0.0
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            optimizer.zero_grad()
            scaled_logits = ts_model(idx)  # (batch, n_tasks) for n_lbs == 1

            bool_masks = masks_t[idx].to(torch.bool)
            masked_logits = scaled_logits.view(-1, n_tasks, n_lbs)
            if n_lbs == 1:
                masked_logits = masked_logits.squeeze(-1)
            masked_logits = masked_logits[bool_masks]
            masked_lbs = labels_t[idx][bool_masks]

            loss_elem = F.binary_cross_entropy_with_logits(masked_logits, masked_lbs, reduction="none")
            m = masks_t[idx]
            loss = torch.sum(loss_elem) / m.sum().clamp_min(1.0)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * m.sum().item()
            epoch_count += m.sum().item()
        loss_curve.append(epoch_loss / max(epoch_count, 1.0))

    with torch.no_grad():
        final_temperature = ts_model.temperature.detach().cpu().numpy().copy()

    return CachedTemperatureFitResult(
        temperature=final_temperature,
        loss_curve=loss_curve,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=bs,
        optimizer_betas=optimizer_betas,
        optimizer_eps=optimizer_eps,
    )


def apply_temperature(logits: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """calibrated_logits = raw_logits / T -- MUBen's TSModel.forward() convention."""
    return logits / temperature[np.newaxis, :]
