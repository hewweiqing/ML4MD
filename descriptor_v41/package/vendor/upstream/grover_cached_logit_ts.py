"""Cache-once temperature-scaling fitting for GROVER, mirroring cached_logit_ts.py's
architecture (see that module's docstring for the "cache once, fit on cached validation
logits only, apply to the same cached test logits" rationale) but adapted to GROVER's
DISTINCT temperature-scaling implementation:

- MUBen's GROVER-specific `muben/model/grover/ts.py::TSModel` (imported here as
  `GROVERTSModel`) holds TWO independent per-task temperature vectors --
  `atom_temperature` and `bond_temperature` -- not one, because GROVER's base model always
  returns a `(atom_logits, bond_logits)` tuple (its dual message-passing branches).
- `TrainerGrover.ts_session()` does NOT override `initialize_optimizer()`, so it uses base
  `Trainer.initialize_optimizer()` verbatim: `AdamW(params, lr=self._status.lr)` with
  PyTorch's plain defaults (betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01) -- unlike
  Uni-Mol, which overrides betas/eps. `params` = `[p for p in self.model.parameters() if
  p.requires_grad]`; since `self.freeze()` was called first, this resolves to exactly
  `[atom_temperature, bond_temperature]`.
- `TrainerGrover.get_loss()` computes `atom_loss + bond_loss` (each the base masked-BCE
  formula, `torch.sum(loss)/masks.sum()`, applied independently to the atom- and
  bond-branch logits) and, because `self._ts_model is not None` during TS fitting, skips
  the distance-loss term entirely -- so the two temperature vectors are optimized by a
  single joint AdamW step over a summed loss, but since each branch's loss only depends on
  its own temperature, this is equivalent to independent per-branch fitting.

This module is used ONLY for the exact-reproduction track; no calibration.py extension-track
code is imported.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

import _pathsetup  # noqa: F401
from muben.model.grover.ts import TSModel as GROVERTSModel


class _CachedDualLogitsReplay(torch.nn.Module):
    """Stand-in 'model' for GROVERTSModel: given a batch of integer row indices, returns the
    corresponding rows of precomputed (atom_logits, bond_logits) -- no real forward pass.
    GROVERTSModel.forward() calls `self.model.eval()` then `self.model(batch)`; both are
    satisfied by this module (nn.Module.eval() is inherited; forward() is overridden below).
    """

    def __init__(self, atom_logits: torch.Tensor, bond_logits: torch.Tensor):
        super().__init__()
        self.register_buffer("_atom_logits", atom_logits)
        self.register_buffer("_bond_logits", bond_logits)

    def forward(self, idx: torch.Tensor):
        return self._atom_logits[idx], self._bond_logits[idx]


@dataclass
class GroverCachedTemperatureFitResult:
    atom_temperature: np.ndarray
    bond_temperature: np.ndarray
    loss_curve: list
    n_epochs: int
    lr: float
    batch_size: int
    optimizer_betas: tuple
    optimizer_eps: float


def _masked_bce_sum_over_mask_sum(logits: torch.Tensor, lbs: torch.Tensor, masks: torch.Tensor, n_tasks: int) -> torch.Tensor:
    """Reproduces base Trainer.get_loss()'s exact classification-branch formula:
    BCEWithLogitsLoss(reduction='none') on masked elements, then sum(loss) / masks.sum()
    (over the WHOLE batch's mask, not just the masked-selected elements) -- matching
    trainer.py:965-966 verbatim (n_lbs == 1 binary classification case).
    """
    bool_masks = masks.to(torch.bool)
    masked_logits = logits.view(-1, n_tasks, 1)[bool_masks].squeeze(-1)
    masked_lbs = lbs[bool_masks]
    loss = F.binary_cross_entropy_with_logits(masked_logits, masked_lbs, reduction="none")
    return torch.sum(loss) / masks.sum()


def fit_temperature_from_cached_logits(
    val_atom_logits: np.ndarray,
    val_bond_logits: np.ndarray,
    val_labels: np.ndarray,
    val_masks: np.ndarray,
    n_tasks: int,
    lr: float = 0.01,
    n_epochs: int = 20,
    batch_size: Optional[int] = None,
    optimizer_betas: tuple = (0.9, 0.999),
    optimizer_eps: float = 1e-8,
    seed: int = 0,
    device: str = "cpu",
) -> GroverCachedTemperatureFitResult:
    """Fits GROVER's native dual-branch TSModel (atom_temperature, bond_temperature) using
    CACHED validation logits for both branches -- one forward pass' worth of data, reused
    across every optimization step, instead of a second live forward pass per epoch as
    native ts_session() would do.

    `optimizer_betas`/`optimizer_eps` default to PyTorch's plain AdamW defaults, matching
    TrainerGrover's un-overridden `initialize_optimizer()` (see module docstring).

    No test data can reach this function -- only `val_*` arrays are accepted.
    """
    if not (val_atom_logits.shape == val_bond_logits.shape == val_labels.shape == val_masks.shape):
        raise ValueError("val_atom_logits, val_bond_logits, val_labels, val_masks must share shape (N, n_tasks)")

    torch.manual_seed(seed)

    atom_logits_t = torch.as_tensor(val_atom_logits, dtype=torch.float32, device=device)
    bond_logits_t = torch.as_tensor(val_bond_logits, dtype=torch.float32, device=device)
    labels_t = torch.as_tensor(val_labels, dtype=torch.float32, device=device)
    masks_t = torch.as_tensor(val_masks, dtype=torch.float32, device=device)

    replay_model = _CachedDualLogitsReplay(atom_logits_t, bond_logits_t)
    ts_model = GROVERTSModel(replay_model, n_tasks)

    optimizer = AdamW(
        [ts_model.atom_temperature, ts_model.bond_temperature],
        lr=lr,
        betas=optimizer_betas,
        eps=optimizer_eps,
    )

    n = atom_logits_t.shape[0]
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
            atom_scaled, bond_scaled = ts_model(idx)

            m = masks_t[idx]
            lbs = labels_t[idx]
            atom_loss = _masked_bce_sum_over_mask_sum(atom_scaled, lbs, m, n_tasks)
            bond_loss = _masked_bce_sum_over_mask_sum(bond_scaled, lbs, m, n_tasks)
            loss = atom_loss + bond_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * m.sum().item()
            epoch_count += m.sum().item()
        loss_curve.append(epoch_loss / max(epoch_count, 1.0))

    with torch.no_grad():
        final_atom_temperature = ts_model.atom_temperature.detach().cpu().numpy().copy()
        final_bond_temperature = ts_model.bond_temperature.detach().cpu().numpy().copy()

    return GroverCachedTemperatureFitResult(
        atom_temperature=final_atom_temperature,
        bond_temperature=final_bond_temperature,
        loss_curve=loss_curve,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=bs,
        optimizer_betas=optimizer_betas,
        optimizer_eps=optimizer_eps,
    )


def apply_temperature_and_average(
    atom_logits: np.ndarray,
    bond_logits: np.ndarray,
    atom_temperature: np.ndarray,
    bond_temperature: np.ndarray,
) -> np.ndarray:
    """Matches TrainerGrover.process_logits()'s exact convention:
    `(sigmoid(atom_logits / atom_T) + sigmoid(bond_logits / bond_T)) / 2`.
    """
    atom_scaled = atom_logits / atom_temperature[np.newaxis, :]
    bond_scaled = bond_logits / bond_temperature[np.newaxis, :]
    atom_probs = 1.0 / (1.0 + np.exp(-atom_scaled))
    bond_probs = 1.0 / (1.0 + np.exp(-bond_scaled))
    return (atom_probs + bond_probs) / 2.0


def raw_probs_from_dual_logits(atom_logits: np.ndarray, bond_logits: np.ndarray) -> np.ndarray:
    """Matches TrainerGrover.process_logits()'s raw (untemperature-scaled) convention:
    `(sigmoid(atom_logits) + sigmoid(bond_logits)) / 2`.
    """
    atom_probs = 1.0 / (1.0 + np.exp(-atom_logits))
    bond_probs = 1.0 / (1.0 + np.exp(-bond_logits))
    return (atom_probs + bond_probs) / 2.0
