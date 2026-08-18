"""Non-scientific synthetic execution checks used by the v17 smoke gate."""

from __future__ import annotations

import copy
import hashlib
import random

import numpy as np
import torch

from .production import assert_temperature_invariance


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class _TwoClassSmokeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU())
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        descriptor = self.encoder(values)
        logits = self.fc(descriptor)
        return torch.nn.functional.log_softmax(logits, dim=1), logits


def _rng_snapshot() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _rng_restore(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _train_two_epochs(device: torch.device, *, interrupt_after_epoch: int | None = None,
        resume: dict | None = None) -> tuple[dict, dict | None]:
    torch.manual_seed(701)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(701)
    model = _TwoClassSmokeModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=2, steps_per_epoch=3,
        pct_start=0.3, anneal_strategy="cos", cycle_momentum=True, div_factor=25, final_div_factor=10000)
    start_epoch, optimizer_steps = 0, 0
    restored_order_state = None
    if resume is not None:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        _rng_restore(resume["rng_state"])
        restored_order_state = resume["data_order_generator_state"]
        start_epoch, optimizer_steps = resume["epoch"] + 1, resume["optimizer_steps"]
    values = torch.tensor([[0.2, -0.1, 0.4], [1.0, 0.5, -0.2], [-0.4, 0.7, 0.1],
        [0.8, -0.3, 0.5], [-0.9, 0.2, 0.3], [0.6, 0.4, -0.7], [0.1, 0.2, 0.3],
        [-0.2, -0.5, 0.9], [0.7, -0.8, 0.2]], dtype=torch.float32, device=device)
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long, device=device)
    checkpoint = None
    for epoch in range(start_epoch, 2):
        generator = np.random.default_rng(900 + epoch)
        if epoch == start_epoch and restored_order_state is not None:
            generator.bit_generator.state = restored_order_state
        order = generator.permutation(len(labels)).tolist()
        model.train()
        for start in range(0, len(order), 4):
            selected = order[start:start + 4]
            log_probabilities, logits = model(values[selected])
            nll = torch.nn.functional.nll_loss(log_probabilities, labels[selected])
            cross_entropy = torch.nn.functional.cross_entropy(logits, labels[selected])
            if log_probabilities.shape != (len(selected), 2) or not torch.allclose(nll, cross_entropy, rtol=1e-6, atol=1e-7):
                raise RuntimeError("synthetic raw-logit/final-batch contract failed")
            optimizer.zero_grad(set_to_none=True)
            nll.backward()
            optimizer.step()
            scheduler.step()
            optimizer_steps += 1
        next_generator = np.random.default_rng(900 + epoch + 1)
        checkpoint = {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()), "epoch": epoch, "optimizer_steps": optimizer_steps,
            "data_order_generator_state": copy.deepcopy(next_generator.bit_generator.state), "rng_state": _rng_snapshot()}
        if interrupt_after_epoch == epoch:
            break
    return {"model_sha256": _state_hash(model.state_dict()), "optimizer_steps": optimizer_steps,
        "scheduler_last_epoch": scheduler.last_epoch}, checkpoint


def run_execution_contracts(device: str = "cuda") -> dict:
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the DelftBlue execution-contract smoke")
    torch.manual_seed(700)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(700)
    control = _TwoClassSmokeModel().to(selected)
    random2 = _TwoClassSmokeModel().to(selected)
    random2.load_state_dict(copy.deepcopy(control.state_dict()))
    initial_control = _state_hash(control.state_dict())
    initial_random2 = _state_hash(random2.state_dict())
    if initial_control != initial_random2:
        raise RuntimeError("paired initial model hashes differ")
    batchnorm_probe = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4)).to(selected)
    batchnorm_probe.eval()
    batchnorm_before = _state_hash(batchnorm_probe.state_dict())
    with torch.no_grad():
        batchnorm_probe(torch.ones((5, 3), device=selected))
    batchnorm_after = _state_hash(batchnorm_probe.state_dict())
    if batchnorm_before != batchnorm_after:
        raise RuntimeError("eval/no_grad descriptor pass changed BatchNorm parameters or buffers")
    full, _ = _train_two_epochs(selected)
    _, checkpoint = _train_two_epochs(selected, interrupt_after_epoch=0)
    resumed, _ = _train_two_epochs(selected, resume=checkpoint)
    if full != resumed or full["optimizer_steps"] != 6 or full["scheduler_last_epoch"] != 6:
        raise RuntimeError(f"uninterrupted/resumed or OneCycleLR contract failed: full={full}, resumed={resumed}")
    rng = _rng_snapshot()
    expected_cpu = torch.rand(4)
    expected_cuda = torch.rand(4, device="cuda").cpu() if torch.cuda.is_available() else None
    _rng_restore(rng)
    if not torch.equal(expected_cpu, torch.rand(4)):
        raise RuntimeError("torch CPU RNG restoration failed")
    if expected_cuda is not None and not torch.equal(expected_cuda, torch.rand(4, device="cuda").cpu()):
        raise RuntimeError("CUDA RNG restoration failed")
    raw = np.asarray([[2.0, -1.0], [-2.0, 3.0], [1.0, 0.0], [-1.0, 2.0]], dtype=np.float64)
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    invariance = assert_temperature_invariance(raw, raw / 2.75, labels)
    return {"status": "passed", "non_scientific_synthetic_only": True,
        "final_batch_size_one_exercised": True, "nll_cross_entropy_equivalence": True,
        "initial_model_sha256": initial_control, "uninterrupted_model_sha256": full["model_sha256"],
        "resumed_model_sha256": resumed["model_sha256"], "optimizer_steps": full["optimizer_steps"],
        "onecycle_scheduler_steps": full["scheduler_last_epoch"], "rng_restoration": ["CPU", "CUDA"],
        "sample_order_generator_restored": True, "batchnorm_state_byte_identical_in_eval": True,
        "temperature_invariance": invariance}
