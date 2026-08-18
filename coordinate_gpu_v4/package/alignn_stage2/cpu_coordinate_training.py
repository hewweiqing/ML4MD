"""Frozen ALIGNN Control/Random2 training with epoch-boundary deterministic resume.

This module never resolves or materializes an outer-test split and contains no
temperature-scaling implementation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import signal
import time
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import dgl
import ijson
import numpy as np
import torch
from alignn.graphs import Graph
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from jarvis.core.atoms import pmg_to_atoms
from matbench.bench import MatbenchBenchmark
from pymatgen.core import Structure
from sklearn.model_selection import train_test_split

from .common import ALIGNN_COMMIT, DATASET_SHA256, FOLDS, SEEDS, append_jsonl, sha256_file, sha256_ids, write_json
from .structure_cache import expected_provenance, load_selected
from .coordinate_cache import load_coordinate_graphs
from .cpu_runtime import assert_model_cpu, configure_cpu_runtime
from .sigma_authorization import require_authorized_sigma

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 32
EPOCHS = 40
JARVIS_VERSION = version("jarvis-tools")
GRAPH_SETTINGS = {"neighbor_strategy": "k-nearest", "cutoff": 8.0, "max_neighbors": 12, "atom_features": "cgcnn", "compute_line_graph": True, "use_canonize": True}
os.environ.setdefault("DGLBACKEND", "pytorch")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    


def model_config() -> ALIGNNConfig:
    return ALIGNNConfig(name="alignn", alignn_layers=4, gcn_layers=4, atom_input_features=92,
        edge_input_features=80, triplet_input_features=40, embedding_features=64,
        hidden_features=256, classification=True, num_classes=2, link="identity")


def model_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parameter_hash(model: torch.nn.Module, *, classifier: bool) -> str:
    values = {name: value for name, value in model.state_dict().items() if name.startswith("fc.") == classifier}
    return model_state_sha256(values)


def component_state(model_or_state, *, classifier: bool) -> dict[str, torch.Tensor]:
    """Clone the complete head or non-head state, including persistent buffers."""
    state = model_or_state.state_dict() if hasattr(model_or_state, "state_dict") else model_or_state
    return {name: value.detach().cpu().contiguous().clone() for name, value in state.items()
        if name.startswith("fc.") == classifier}


def states_byte_identical(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    if set(left) != set(right):
        return False
    return all(left[name].dtype == right[name].dtype and left[name].shape == right[name].shape
        and torch.equal(left[name].detach().cpu(), right[name].detach().cpu()) for name in left)


def capture_rng_state() -> dict:
    return {"python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state()}


def restore_rng_state(state: dict) -> None:
    required = {"python_random_state", "numpy_random_state", "torch_cpu_rng_state"}
    if set(state) != required:
        raise RuntimeError("resume checkpoint does not contain the complete Python/NumPy/torch-CPU RNG state")
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    


def iter_batches(order: list[int]):
    for start in range(0, len(order), BATCH_SIZE):
        yield order[start:start + BATCH_SIZE]


def graph_batch(atom_graphs, line_graphs, indices):
    return dgl.batch([atom_graphs[i] for i in indices]).to("cpu"), dgl.batch([line_graphs[i] for i in indices]).to("cpu"), None


def _two_class_matrix(values: torch.Tensor, name: str) -> torch.Tensor:
    if values.ndim == 1 and values.numel() == 2:
        values = values.reshape(1, 2)
    if values.ndim != 2 or values.shape[1] != 2:
        raise RuntimeError(f"{name} must have shape [batch,2], including a final batch of size one")
    return values


def assert_native_logit_contract(log_probabilities: torch.Tensor, logits: torch.Tensor,
        labels: torch.Tensor | None = None) -> None:
    expected = torch.nn.functional.log_softmax(logits, dim=1)
    if log_probabilities.shape != logits.shape or not torch.allclose(log_probabilities, expected, rtol=1e-6, atol=1e-7):
        raise RuntimeError("ALIGNN forward output is not LogSoftmax(ALIGNN.fc pre-LogSoftmax logits)")
    if labels is not None:
        labels = labels.reshape(-1)
        nll = torch.nn.functional.nll_loss(log_probabilities, labels)
        cross_entropy = torch.nn.functional.cross_entropy(logits, labels)
        if not torch.allclose(nll, cross_entropy, rtol=1e-6, atol=1e-7):
            raise RuntimeError("NLLLoss(LogSoftmax(z), y) != CrossEntropyLoss(z, y)")


def forward_raw(model, batch, labels: torch.Tensor | None = None):
    captured = []
    hook = model.fc.register_forward_hook(lambda _module, _inputs, output: captured.append(output))
    try:
        log_probabilities = model(batch)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError("ALIGNN.fc must execute exactly once per classification forward pass")
    logits = _two_class_matrix(captured[0], "ALIGNN.fc output")
    log_probabilities = _two_class_matrix(log_probabilities, "ALIGNN forward output")
    assert_native_logit_contract(log_probabilities, logits, labels)
    return log_probabilities, logits


def evaluate_raw(model, atom_graphs, line_graphs, labels):
    model.eval()
    output = []
    with torch.no_grad():
        for indices in iter_batches(list(range(len(labels)))):
            batch_labels = torch.tensor([labels[index] for index in indices], device="cpu")
            _, logits = forward_raw(model, graph_batch(atom_graphs, line_graphs, indices), batch_labels)
            output.append(logits.detach().cpu())
    return torch.cat(output).numpy()


def validation_nll(logits: np.ndarray, labels: list[int]) -> float:
    return float(torch.nn.functional.cross_entropy(torch.from_numpy(logits), torch.tensor(labels)).item())


def validate_frozen_files() -> None:
    frozen = json.loads((PACKAGE_ROOT / "STAGE_2_AMENDMENT_HASHES.json").read_text(encoding="utf-8"))
    for item in frozen["entries"]:
        path = PACKAGE_ROOT / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"frozen scientific file hash mismatch: {item['path']}")


def load_inner_split(dataset: Path, fold: int, seed: int):
    benchmark = MatbenchBenchmark(autoload=False, subset=["matbench_mp_is_metal"])
    task = next(iter(benchmark.tasks))
    fold_key = task.folds_map[fold]
    outer_training_ids = list(task.validation[fold_key].train)
    allowed = set(outer_training_ids)
    labels = {}
    with gzip.open(dataset, "rb") as stream:
        row_index = -1
        for prefix, event, value in ijson.parse(stream):
            if prefix == "data.item.item" and event == "boolean":
                row_index += 1
                structure_id = f"mb-mp-is-metal-{row_index + 1:06d}"
                if structure_id in allowed:
                    labels[structure_id] = int(value)
    if set(labels) != allowed:
        raise RuntimeError("outer-training ID/label coverage mismatch")
    train_ids, validation_ids = train_test_split(outer_training_ids, test_size=0.1, random_state=seed,
        shuffle=True, stratify=[labels[item] for item in outer_training_ids])
    audit = json.loads((PACKAGE_ROOT / "ALL_25_SPLIT_HASHES.json").read_text(encoding="utf-8"))
    expected = next(item for item in audit["cells"] if item["fold"] == fold and item["seed"] == seed)
    if (sha256_ids(train_ids) != expected["inner_train_ids_sha256"]
            or sha256_ids(validation_ids) != expected["inner_validation_ids_sha256"]
            or sha256_ids(labels[item] for item in validation_ids) != expected["inner_validation_labels_sha256"]):
        raise RuntimeError("frozen fold/seed split hash mismatch")
    return train_ids, validation_ids, [labels[item] for item in train_ids], [labels[item] for item in validation_ids]


def cache_expected_provenance(dataset: Path) -> dict:
    if sha256_file(dataset) != DATASET_SHA256:
        raise RuntimeError("official dataset SHA-256 mismatch before cache access")
    return expected_provenance(GRAPH_SETTINGS, jarvis_version=JARVIS_VERSION, dgl_version=dgl.__version__,
        release_identity="1c15f877cee4eb6eebfd096962c6a793d0d3f14385287a6a9385e3adce05bd7d")


def build_or_load_graphs(dataset: Path, ids: list[str], cache_dir: Path, split: str, *, return_stats: bool = False):
    if not (Path(cache_dir) / "STRUCTURE_CACHE_MANIFEST.json").is_file():
        raise RuntimeError(f"verified structure-level graph cache is required before {split}")
    return load_selected(Path(cache_dir), ids, cache_expected_provenance(dataset), return_stats=return_stats)


def create_start_states(train_atoms, train_lines, seed: int, work_dir: Path):
    base_path, random2_path = work_dir / "paired_base.pt", work_dir / "random2_warmup_only.pt"
    if base_path.exists() != random2_path.exists():
        raise RuntimeError("incomplete paired start-state artifacts")
    if base_path.exists():
        base_payload = torch.load(base_path, map_location="cpu")
        random2_payload = torch.load(random2_path, map_location="cpu")
        if base_payload.get("integrity_schema") != 2 or random2_payload.get("integrity_schema") != 2:
            raise RuntimeError("legacy paired start states lack the v17 descriptor/BatchNorm integrity proof")
        base_state, random2_state = base_payload["model"], random2_payload["model"]
        if not states_byte_identical(component_state(base_state, classifier=False),
                component_state(random2_state, classifier=False)):
            raise RuntimeError("resumed paired start states have non-identical encoders")
        return base_state, random2_state
    seed_all(seed)
    base = ALIGNN(model_config()).to("cpu")
    initialized_state = deepcopy(base.state_dict())
    encoder_before_descriptors = component_state(initialized_state, classifier=False)
    descriptors = []
    base.eval()
    if base.training:
        raise RuntimeError("Random2 descriptor extraction must run with the full ALIGNN model in eval mode")
    with torch.no_grad():
        for indices in iter_batches(list(range(len(train_atoms)))):
            captured = []
            hook = base.fc.register_forward_hook(lambda _module, inputs, _output: captured.append(inputs[0]))
            base(graph_batch(train_atoms, train_lines, indices))
            hook.remove()
            descriptor_batch = captured[-1]
            if descriptor_batch.ndim == 1:
                descriptor_batch = descriptor_batch.reshape(1, -1)
            if descriptor_batch.ndim != 2 or descriptor_batch.shape[1] != base.fc.in_features:
                raise RuntimeError("pooled descriptor batch must remain [batch, fc.in_features], including batch size one")
            descriptors.append(descriptor_batch.detach())
    descriptors = torch.cat(descriptors)
    encoder_after_descriptors = component_state(base, classifier=False)
    if not states_byte_identical(encoder_before_descriptors, encoder_after_descriptors):
        raise RuntimeError("Random2 descriptor extraction modified non-head parameters or BatchNorm buffers")
    base_state = deepcopy(initialized_state)
    initial_sha256 = model_state_sha256(base_state)
    write_json(work_dir / "paired_initial_state.json", {"seed": seed, "initial_state_sha256": initial_sha256,
        "control_initial_state_sha256": initial_sha256, "random2_pre_warmup_initial_state_sha256": initial_sha256,
        "descriptor_tensor": "input[0] captured by forward hook on ALIGNN.fc",
        "descriptor_extraction_mode": "eval", "descriptor_extraction_no_grad": True,
        "non_head_state_includes_buffers": True,
        "encoder_state_before_descriptors_sha256": model_state_sha256(encoder_before_descriptors),
        "encoder_state_after_descriptors_sha256": model_state_sha256(encoder_after_descriptors),
        "encoder_state_byte_identical_after_descriptors": True,
        "descriptor_shape": list(descriptors.shape), "one_descriptor_per_training_structure": len(descriptors) == len(train_atoms),
        "descriptor_dimension": int(descriptors.shape[1]), "classification_head_input_dimension": int(base.fc.in_features)})
    def warm_once():
        random2 = ALIGNN(model_config()).to("cpu")
        random2.load_state_dict(base_state)
        encoder_before_state = component_state(random2, classifier=False)
        classifier_before_state = component_state(random2, classifier=True)
        encoder_before, classifier_before = model_state_sha256(encoder_before_state), model_state_sha256(classifier_before_state)
        for name, parameter in random2.named_parameters():
            parameter.requires_grad_(name.startswith("fc."))
        mean, std = descriptors.mean(0), descriptors.std(0, unbiased=False).clamp_min(1e-6)
        generator = torch.Generator(device="cpu").manual_seed(seed + 10000)
        holdout_generator = torch.Generator(device="cpu").manual_seed(seed + 10001)
        synthetic_count = int(os.getenv("ALIGNN_RANDOM2_SYNTHETIC_COUNT", "3000"))
        warmup_steps = int(os.getenv("ALIGNN_RANDOM2_WARMUP_STEPS", "938"))
        if synthetic_count <= 0 or synthetic_count % 2:
            raise RuntimeError("Random2 synthetic count must be positive and even")
        synthetic = mean + std * torch.randn((synthetic_count, descriptors.shape[1]), generator=generator, device="cpu")
        labels = torch.tensor([0, 1] * (synthetic_count // 2), device="cpu")
        permutation = torch.randperm(synthetic_count, generator=generator, device="cpu")
        synthetic, labels = synthetic[permutation], labels[permutation]
        optimizer = torch.optim.AdamW(random2.fc.parameters(), lr=1e-4, weight_decay=0.0)
        cursor = 0
        for _ in range(warmup_steps):
            if cursor + 128 > synthetic_count:
                permutation = torch.randperm(synthetic_count, generator=generator, device="cpu")
                cursor = 0
            indices, cursor = permutation[cursor:cursor + 128], cursor + 128
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(random2.fc(synthetic[indices]), labels[indices]).backward()
            optimizer.step()
        heldout = mean + std * torch.randn((synthetic_count, descriptors.shape[1]), generator=holdout_generator, device="cpu")
        with torch.no_grad():
            probabilities = torch.softmax(random2.fc(heldout), dim=1)
            entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-15))).sum(1)
        diagnostics = {"finite": bool(torch.isfinite(probabilities).all()),
            "mean_positive_probability": float(probabilities[:, 1].mean().cpu()),
            "mean_entropy": float(entropy.mean().cpu()), "max_entropy": float(entropy.max().cpu()),
            "encoder_unchanged": states_byte_identical(encoder_before_state, component_state(random2, classifier=False)),
            "classifier_changed": not states_byte_identical(classifier_before_state, component_state(random2, classifier=True)),
            "encoder_hash_before": encoder_before, "encoder_hash_after": parameter_hash(random2, classifier=False),
            "classifier_hash_before": classifier_before, "classifier_hash_after": parameter_hash(random2, classifier=True),
            "encoder_state_scope": "complete non-head state_dict including BatchNorm buffers",
            "heldout_descriptors_optimized_directly": False, "heldout_descriptor_seed": seed + 10001}
        passed = bool(diagnostics["finite"] and 0.45 <= diagnostics["mean_positive_probability"] <= 0.55
            and diagnostics["mean_entropy"] >= 0.68 and diagnostics["max_entropy"] <= math.log(2) + 1e-6
            and diagnostics["encoder_unchanged"] and diagnostics["classifier_changed"])
        return random2, diagnostics, passed
    random2, diagnostics, passed = warm_once()
    replay_model, replay_diagnostics, replay_passed = warm_once()
    diagnostics["deterministic_replay"] = model_state_sha256(random2.state_dict()) == model_state_sha256(replay_model.state_dict())
    diagnostics["replay_diagnostics_equal"] = diagnostics["mean_positive_probability"] == replay_diagnostics["mean_positive_probability"] and diagnostics["mean_entropy"] == replay_diagnostics["mean_entropy"]
    if not passed or not replay_passed or not diagnostics["deterministic_replay"] or not diagnostics["replay_diagnostics_equal"]:
        raise RuntimeError("Random2 mechanical gate failed")
    random2_state = deepcopy(random2.state_dict())
    if not states_byte_identical(component_state(base_state, classifier=False),
            component_state(random2_state, classifier=False)):
        raise RuntimeError("Control and Random2 encoders are not byte-identical before supervised training")
    torch.save({"model": base_state, "seed": seed, "integrity_schema": 2,
        "native_logit_source": "ALIGNN.fc output before LogSoftmax"}, base_path)
    torch.save({"model": random2_state, "seed": seed, "diagnostics": diagnostics, "integrity_schema": 2,
        "non_head_state_byte_identical_to_control": True}, random2_path)
    write_json(work_dir / "random2_diagnostics.json", diagnostics)
    return base_state, random2_state


def create_coordinate_start_states(fold: int, seed: int, work_dir: Path, coordinate_cache_root: Path,
        authorization):
    base_path, random2_path = work_dir / "paired_cpu_base.pt", work_dir / "coordinate_warmup_only.pt"
    if base_path.exists() != random2_path.exists():
        raise RuntimeError("incomplete paired CPU start-state artifacts")
    if base_path.exists():
        base_payload, random2_payload = torch.load(base_path, map_location="cpu"), torch.load(random2_path, map_location="cpu")
        if (base_payload.get("authorization_sha256") != authorization.authorization_sha256
                or random2_payload.get("authorization_sha256") != authorization.authorization_sha256):
            raise RuntimeError("paired start state was created under a different sigma authorization")
        if not states_byte_identical(component_state(base_payload["model"], classifier=False),
                component_state(random2_payload["model"], classifier=False)):
            raise RuntimeError("resumed CPU paired encoders differ")
        return base_payload["model"], random2_payload["model"]
    seed_all(seed)
    base = ALIGNN(model_config()).to(torch.device("cpu")); assert_model_cpu(base)
    initialized = deepcopy(base.state_dict())
    initial_hash = model_state_sha256(initialized)
    coordinate_cell = coordinate_cache_root / f"fold_{fold}" / f"seed_{seed}"
    atom_graphs, line_graphs, records = load_coordinate_graphs(coordinate_cell, authorization)
    if len(records) != 3000 or sum(int(row["random_label"]) for row in records) != 1500:
        raise RuntimeError("coordinate warm-up cache must contain exactly 3000 balanced records")
    if any(row.get("true_label_used") is not False for row in records):
        raise RuntimeError("coordinate warm-up provenance indicates true-label use")
    non_head_before = component_state(base, classifier=False)
    base.eval(); descriptors = []
    if base.training:
        raise RuntimeError("coordinate descriptor extraction must use eval mode")
    with torch.no_grad():
        for indices in iter_batches(list(range(len(records)))):
            captured = []
            hook = base.fc.register_forward_hook(lambda _module, inputs, _output: captured.append(inputs[0]))
            base(graph_batch(atom_graphs, line_graphs, indices)); hook.remove()
            values = captured[-1]
            if values.ndim == 1: values = values.reshape(1, -1)
            descriptors.append(values.detach().cpu())
    descriptors = torch.cat(descriptors)
    if not torch.isfinite(descriptors).all() or not states_byte_identical(non_head_before, component_state(base, classifier=False)):
        raise RuntimeError("coordinate descriptor extraction changed encoder/buffers or produced non-finite values")
    random2 = ALIGNN(model_config()).to(torch.device("cpu")); random2.load_state_dict(initialized); assert_model_cpu(random2)
    head_before, encoder_before = component_state(random2, classifier=True), component_state(random2, classifier=False)
    for name, parameter in random2.named_parameters(): parameter.requires_grad_(name.startswith("fc."))
    labels = torch.tensor([int(row["random_label"]) for row in records], dtype=torch.long, device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(seed + 280000)
    optimizer = torch.optim.AdamW(random2.fc.parameters(), lr=1e-4, weight_decay=0.0)
    history, cursor = [], 0; permutation = torch.randperm(3000, generator=generator)
    for step in range(938):
        if cursor + 128 > 3000:
            permutation = torch.randperm(3000, generator=generator); cursor = 0
        indices, cursor = permutation[cursor:cursor + 128], cursor + 128
        optimizer.zero_grad(set_to_none=True); logits = random2.fc(descriptors[indices])
        loss = torch.nn.functional.cross_entropy(logits, labels[indices]); loss.backward(); optimizer.step()
        if step in {0, 9, 99, 499, 937}:
            with torch.no_grad():
                p = torch.softmax(logits, dim=1)
                history.append({"step": step + 1, "loss": float(loss),
                    "entropy": float((-(p * p.clamp_min(1e-15).log()).sum(1)).mean()),
                    "mean_max_probability": float(p.max(1).values.mean()),
                    "prediction_positive_fraction": float((p[:, 1] > p[:, 0]).float().mean())})
    if not states_byte_identical(encoder_before, component_state(random2, classifier=False)):
        raise RuntimeError("coordinate warm-up changed encoder parameters or BatchNorm buffers")
    if states_byte_identical(head_before, component_state(random2, classifier=True)):
        raise RuntimeError("coordinate warm-up did not change fc.weight/fc.bias")
    changed = {name for name in initialized if not torch.equal(initialized[name], random2.state_dict()[name].cpu())}
    if not changed or not changed.issubset({"fc.weight", "fc.bias"}):
        raise RuntimeError(f"coordinate warm-up changed prohibited state: {sorted(changed)}")
    random2_state = deepcopy(random2.state_dict())
    _atomic_torch_save({"model": initialized, "initial_state_sha256": initial_hash,
        "authorization_sha256": authorization.authorization_sha256}, base_path)
    _atomic_torch_save({"model": random2_state, "initial_state_sha256": initial_hash,
        "authorization_sha256": authorization.authorization_sha256}, random2_path)
    write_json(work_dir / "coordinate_warmup_diagnostics.json", {"status": "passed", "fold": fold, "seed": seed,
        "records": 3000, "balanced_labels": True, "optimizer_steps": 938, "batch_size": 128,
        "initial_full_model_sha256": initial_hash, "encoder_byte_identical": True,
        "changed_state_keys": sorted(changed), "head_before_sha256": model_state_sha256(head_before),
        "head_after_sha256": model_state_sha256(component_state(random2_state, classifier=True)),
        "descriptor_shape": list(descriptors.shape), "descriptor_nonfinite_count": int((~torch.isfinite(descriptors)).sum()),
        "history": history, "true_labels_used": False, "execution_device": "cpu",
        "authorization_sha256": authorization.authorization_sha256})
    seed_all(seed)
    return initialized, random2_state


_STOP_REQUESTED = False


def _request_stop(_signal, _frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _atomic_torch_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary); os.replace(temporary, path)


def train_branch_cpu(branch: str, start_state, train_atoms, train_lines, train_labels, train_ids,
        validation_atoms, validation_lines, validation_labels, validation_ids, seed: int, work_dir: Path,
        integrity: dict):
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    invocation_started = time.monotonic()
    signal.signal(signal.SIGTERM, _request_stop)
    if hasattr(signal, "SIGUSR1"): signal.signal(signal.SIGUSR1, _request_stop)
    branch_dir = work_dir / branch; branch_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = branch_dir / "last.pt", branch_dir / "best.pt"
    seed_all(seed); model = ALIGNN(model_config()).to(torch.device("cpu")); model.load_state_dict(start_state); assert_model_cpu(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    steps_per_epoch = math.ceil(len(train_labels) / BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch, pct_start=0.3, anneal_strategy="cos", cycle_momentum=True,
        div_factor=25, final_div_factor=10000)
    epoch = batch_position = optimizer_steps = 0; history = []; best_nll = float("inf")
    epoch_loss_sum, epoch_sample_count = 0.0, 0
    elapsed_runtime_seconds = 0.0
    order = np.random.default_rng(seed).permutation(len(train_labels)).tolist()
    if last_path.is_file():
        checkpoint = torch.load(last_path, map_location="cpu")
        if checkpoint.get("integrity") != integrity: raise RuntimeError("resume integrity/package/config/cache mismatch")
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"]); restore_rng_state(checkpoint["rng_state"])
        epoch, batch_position, optimizer_steps = checkpoint["epoch"], checkpoint["batch_position"], checkpoint["optimizer_steps"]
        history, best_nll, order = checkpoint["history"], checkpoint["best_nll"], checkpoint["order"]
        epoch_loss_sum = float(checkpoint.get("epoch_loss_sum", 0.0))
        epoch_sample_count = int(checkpoint.get("epoch_sample_count", 0))
        elapsed_runtime_seconds = float(checkpoint.get("elapsed_runtime_seconds", 0.0))
    while epoch < EPOCHS:
        batches = list(iter_batches(order)); model.train()
        for position in range(batch_position, len(batches)):
            indices = batches[position]; optimizer.zero_grad(set_to_none=True)
            labels = torch.tensor([train_labels[index] for index in indices], dtype=torch.long, device="cpu")
            log_probabilities, _ = forward_raw(model, graph_batch(train_atoms, train_lines, indices), labels)
            loss = torch.nn.functional.nll_loss(log_probabilities, labels); loss.backward(); optimizer.step(); scheduler.step()
            optimizer_steps += 1; epoch_loss_sum += float(loss.detach()) * len(indices); epoch_sample_count += len(indices)
            payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "epoch": epoch, "batch_position": position + 1, "order": order, "history": history,
                "best_nll": best_nll, "optimizer_steps": optimizer_steps, "epoch_loss_sum": epoch_loss_sum,
                "epoch_sample_count": epoch_sample_count,
                "elapsed_runtime_seconds": elapsed_runtime_seconds + time.monotonic() - invocation_started,
                "rng_state": capture_rng_state(), "integrity": integrity}
            if _STOP_REQUESTED:
                _atomic_torch_save(payload, last_path)
                write_json(work_dir / "INCOMPLETE_RESUME_REQUIRED.json", {"status": "incomplete_resume_required",
                    "branch": branch, "epoch": epoch, "next_batch_position": position + 1,
                    "optimizer_steps": optimizer_steps, "checkpoint_sha256": sha256_file(last_path)})
                return history, False
        logits = evaluate_raw(model, validation_atoms, validation_lines, validation_labels)
        nll = validation_nll(logits, validation_labels)
        history.append({"epoch": epoch, "ordered_train_ids_sha256": sha256_ids(train_ids[index] for index in order),
            "train_nll": epoch_loss_sum / epoch_sample_count, "validation_nll": nll,
            "optimizer_steps": optimizer_steps, "scheduler_steps": optimizer_steps})
        if nll < best_nll:
            best_nll = nll; _atomic_torch_save({"model": model.state_dict(), "epoch": epoch, "validation_nll": nll}, best_path)
        epoch += 1; batch_position = 0; epoch_loss_sum, epoch_sample_count = 0.0, 0
        order = np.random.default_rng(seed + epoch).permutation(len(train_labels)).tolist() if epoch < EPOCHS else []
        _atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "batch_position": 0, "order": order, "history": history, "best_nll": best_nll,
            "optimizer_steps": optimizer_steps, "epoch_loss_sum": 0.0, "epoch_sample_count": 0,
            "elapsed_runtime_seconds": elapsed_runtime_seconds + time.monotonic() - invocation_started,
            "rng_state": capture_rng_state(), "integrity": integrity}, last_path)
        write_json(branch_dir / "history.json", history)
    if optimizer_steps != EPOCHS * steps_per_epoch: raise RuntimeError("exact OneCycle optimizer/scheduler step contract failed")
    best = torch.load(best_path, map_location="cpu"); model.load_state_dict(best["model"])
    logits = evaluate_raw(model, validation_atoms, validation_lines, validation_labels)
    reconstructed = ALIGNN(model_config()).to(torch.device("cpu")); reconstructed.load_state_dict(
        torch.load(best_path, map_location="cpu")["model"]); assert_model_cpu(reconstructed)
    reconstructed_logits = evaluate_raw(reconstructed, validation_atoms, validation_lines, validation_labels)
    reconstruction_max_abs_diff = float(np.max(np.abs(logits - reconstructed_logits)))
    if reconstruction_max_abs_diff > 1e-7:
        raise RuntimeError("selected CPU checkpoint reconstruction gate failed")
    np.savez_compressed(branch_dir / "validation_raw_logits.npz", structure_ids=np.asarray(validation_ids),
        labels=np.asarray(validation_labels, dtype=np.int64), logits=logits,
        sample_order_index=np.arange(len(validation_ids), dtype=np.int64),
        logit_source=np.asarray("ALIGNN.fc output before LogSoftmax"), logit_columns=np.asarray(["z_0", "z_1"]))
    write_json(branch_dir / "checkpoint_provenance.json", {"branch": branch, "selected_epoch": best["epoch"],
        "validation_nll": best["validation_nll"], "checkpoint_sha256": sha256_file(best_path),
        "validation_ids_sha256": sha256_ids(validation_ids),
        "validation_labels_sha256": sha256_ids(validation_labels),
        "checkpoint_reconstruction_max_abs_diff": reconstruction_max_abs_diff,
        "optimizer_steps": optimizer_steps, "expected_optimizer_steps": EPOCHS * steps_per_epoch,
        "elapsed_runtime_seconds": elapsed_runtime_seconds + time.monotonic() - invocation_started,
        "execution_device": "cpu", "resume_contract": "batch-position exact on same CPU architecture", **integrity})
    return history, True


def train_branch(branch: str, start_state, train_atoms, train_lines, train_labels, train_ids,
        validation_atoms, validation_lines, validation_labels, validation_ids, seed: int, work_dir: Path):
    branch_dir = work_dir / branch
    branch_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = branch_dir / "last.pt", branch_dir / "best.pt"
    seed_all(seed)
    model = ALIGNN(model_config()).to("cpu")
    model.load_state_dict(start_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    steps_per_epoch = math.ceil(len(train_labels) / BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch, pct_start=0.3, anneal_strategy="cos", cycle_momentum=True,
        div_factor=25, final_div_factor=10000)
    history, start_epoch, best_nll, optimizer_steps = [], 0, float("inf"), 0
    restored_order_generator_state = None
    if last_path.is_file():
        checkpoint = torch.load(last_path, map_location="cpu")
        required_resume = {"rng_state", "data_order_generator_state", "optimizer_steps", "steps_per_epoch"}
        if not required_resume.issubset(checkpoint):
            raise RuntimeError("legacy or incomplete resume checkpoint lacks deterministic v17 state")
        if checkpoint["steps_per_epoch"] != steps_per_epoch:
            raise RuntimeError("resume checkpoint OneCycleLR steps_per_epoch mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng_state(checkpoint["rng_state"])
        restored_order_generator_state = checkpoint["data_order_generator_state"]
        history, start_epoch, best_nll = checkpoint["history"], checkpoint["epoch"] + 1, checkpoint["best_nll"]
        optimizer_steps = int(checkpoint["optimizer_steps"])
        if optimizer_steps != start_epoch * steps_per_epoch:
            raise RuntimeError("resume checkpoint optimizer/OneCycleLR step count mismatch")
    for epoch in range(start_epoch, EPOCHS):
        data_order_generator = np.random.default_rng(seed + epoch)
        if epoch == start_epoch and restored_order_generator_state is not None:
            data_order_generator.bit_generator.state = restored_order_generator_state
        order = data_order_generator.permutation(len(train_labels)).tolist()
        seed_all(seed)
        model.train()
        total_loss = 0.0
        for indices in iter_batches(order):
            optimizer.zero_grad(set_to_none=True)
            labels = torch.tensor([train_labels[index] for index in indices], device="cpu")
            log_probabilities, _ = forward_raw(model, graph_batch(train_atoms, train_lines, indices), labels)
            loss = torch.nn.functional.nll_loss(log_probabilities, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer_steps += 1
            total_loss += float(loss.detach().cpu()) * len(indices)
        logits = evaluate_raw(model, validation_atoms, validation_lines, validation_labels)
        nll = validation_nll(logits, validation_labels)
        history.append({"epoch": epoch, "ordered_train_ids_sha256": sha256_ids(train_ids[index] for index in order),
            "train_nll": total_loss / len(train_labels), "train_nll_aggregation": "sample_weighted",
            "validation_nll": nll, "validation_nll_aggregation": "all_samples_single_cross_entropy",
            "lr": scheduler.get_last_lr()[0], "optimizer_steps": optimizer_steps})
        if nll < best_nll:
            best_nll = nll
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation_nll": nll}, best_path)
        next_order_generator = np.random.default_rng(seed + epoch + 1)
        next_order_generator_state = deepcopy(next_order_generator.bit_generator.state)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "history": history, "best_nll": best_nll, "optimizer_steps": optimizer_steps,
            "steps_per_epoch": steps_per_epoch, "data_order_generator_state": next_order_generator_state,
            "rng_state": capture_rng_state()}, last_path)
        write_json(branch_dir / "history.json", history)
    expected_optimizer_steps = EPOCHS * steps_per_epoch
    if optimizer_steps != expected_optimizer_steps:
        raise RuntimeError(f"OneCycleLR optimizer-step count mismatch: {optimizer_steps} != {expected_optimizer_steps}")
    best = torch.load(best_path, map_location="cpu")
    model.load_state_dict(best["model"])
    logits = evaluate_raw(model, validation_atoms, validation_lines, validation_labels)
    reconstructed = ALIGNN(model_config()).to("cpu")
    reconstructed.load_state_dict(torch.load(best_path, map_location="cpu")["model"])
    reconstructed_logits = evaluate_raw(reconstructed, validation_atoms, validation_lines, validation_labels)
    reconstruction_max_abs_diff = float(np.max(np.abs(logits - reconstructed_logits)))
    if reconstruction_max_abs_diff > 1e-7:
        raise RuntimeError("selected checkpoint reconstruction gate failed")
    np.savez_compressed(branch_dir / "validation_raw_logits.npz", structure_ids=np.asarray(validation_ids),
        labels=np.asarray(validation_labels, dtype=np.int64), logits=logits,
        sample_order_index=np.arange(len(validation_ids), dtype=np.int64),
        logit_source=np.asarray("ALIGNN.fc output before LogSoftmax"),
        logit_columns=np.asarray(["z_0", "z_1"]))
    write_json(branch_dir / "checkpoint_provenance.json", {"branch": branch, "selected_epoch": best["epoch"],
        "validation_nll": best["validation_nll"], "checkpoint_sha256": sha256_file(best_path),
        "validation_ids_sha256": sha256_ids(validation_ids), "validation_labels_sha256": sha256_ids(validation_labels),
        "checkpoint_reconstruction_max_abs_diff": reconstruction_max_abs_diff,
        "native_logit_source": "ALIGNN.fc output before LogSoftmax", "native_logit_columns": ["z_0", "z_1"],
        "nll_cross_entropy_equivalence_asserted_during_every_forward": True,
        "optimizer_steps": optimizer_steps, "expected_optimizer_steps": expected_optimizer_steps,
        "resume_state": ["Python RNG", "NumPy RNG", "torch CPU RNG", "sample-order generator"]})
    return history


def main(argv=None) -> int:
    global BATCH_SIZE, EPOCHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=FOLDS)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--condition", choices=("control", "random2", "paired"), default="paired")
    args = parser.parse_args(argv)
    authorization = require_authorized_sigma()
    runtime = configure_cpu_runtime()
    validate_frozen_files()
    dataset = Path(args.dataset)
    if sha256_file(dataset) != DATASET_SHA256:
        raise RuntimeError("official dataset SHA-256 mismatch")
    work_dir = Path(args.work_root) / f"fold_{args.fold}" / f"seed_{args.seed}"
    if (work_dir / "COMPLETE.json").exists():
        complete = json.loads((work_dir / "COMPLETE.json").read_text(encoding="utf-8"))
        if (complete.get("status"), complete.get("fold"), complete.get("seed"), complete.get("execution_device")) \
                == ("complete", args.fold, args.seed, "cpu"):
            print(f"Verified CPU COMPLETE.json exists; skipping fold={args.fold} seed={args.seed}")
            return 0
        raise RuntimeError("invalid COMPLETE.json present; quarantine this cell before resuming")
    work_dir.mkdir(parents=True, exist_ok=True)
    append_jsonl(work_dir / "RUN_MANIFEST_HISTORY.jsonl", {"event": "TRAINING_INVOCATION_STARTED", "fold": args.fold,
        "seed": args.seed, "condition": args.condition, "outer_test_accessed": False, "slurm_job_id": os.getenv("SLURM_JOB_ID"), "time": time.time()})
    train_ids, validation_ids, train_labels, validation_labels = load_inner_split(dataset, args.fold, args.seed)
    smoke = os.getenv("ALIGNN_NON_PRIMARY_SMOKE") == "1"
    if smoke:
        BATCH_SIZE, EPOCHS = 8, 1
        train_ids, train_labels = train_ids[:64], train_labels[:64]
        validation_ids, validation_labels = validation_ids[:32], validation_labels[:32]
    write_json(work_dir / "split_provenance.json", {"fold": args.fold, "seed": args.seed,
        "split_method": "train_test_split(test_size=0.1, stratify=y, shuffle=True, random_state=seed)",
        "non_primary_smoke": smoke,
        "inner_train_ids_sha256": sha256_ids(train_ids), "inner_validation_ids_sha256": sha256_ids(validation_ids),
        "inner_train_positive_fraction": float(np.mean(train_labels)),
        "inner_validation_labels_sha256": sha256_ids(validation_labels),
        "outer_test_accessed": False})
    cache_root = Path(os.getenv("ALIGNN_GRAPH_CACHE_ROOT", str(Path(args.work_root) / "shared_graph_cache")))
    cache = cache_root
    train_atoms, train_lines = build_or_load_graphs(dataset, train_ids, cache, "inner_train")
    validation_atoms, validation_lines = build_or_load_graphs(dataset, validation_ids, cache, "inner_validation")
    coordinate_cache_root = Path(os.environ["ALIGNN_COORDINATE_CACHE_ROOT"])
    base_state, random2_state = create_coordinate_start_states(args.fold, args.seed, work_dir,
        coordinate_cache_root, authorization)
    if model_state_sha256(base_state) == model_state_sha256(random2_state):
        raise RuntimeError("Random2 head warm-up did not change the paired model state")
    if not states_byte_identical(component_state(base_state, classifier=False),
            component_state(random2_state, classifier=False)):
        raise RuntimeError("paired Control/Random2 non-head state mismatch before supervised training")
    histories = {}
    if args.condition in ("control", "paired"):
        integrity = {"package_aggregate_sha256": json.loads((PACKAGE_ROOT / "PACKAGE_MANIFEST.json").read_text())["aggregate_sha256"],
            "coordinate_authorization_sha256": authorization.authorization_sha256,
            "coordinate_config_sha256": sha256_file(PACKAGE_ROOT / "COORDINATE_NOISE_CONFIG.json"),
            "clean_cache_manifest_sha256": sha256_file(cache_root / "STRUCTURE_CACHE_MANIFEST.json"),
            "coordinate_cache_manifest_sha256": sha256_file(coordinate_cache_root / f"fold_{args.fold}" / f"seed_{args.seed}" / "COORDINATE_CACHE_MANIFEST.json")}
        histories["control"], complete = train_branch_cpu("control", base_state, train_atoms, train_lines, train_labels, train_ids,
            validation_atoms, validation_lines, validation_labels, validation_ids, args.seed, work_dir, integrity)
        if not complete: return 75
    if args.condition in ("random2", "paired"):
        if "integrity" not in locals():
            integrity = {"package_aggregate_sha256": json.loads((PACKAGE_ROOT / "PACKAGE_MANIFEST.json").read_text())["aggregate_sha256"],
                "coordinate_authorization_sha256": authorization.authorization_sha256,
                "coordinate_config_sha256": sha256_file(PACKAGE_ROOT / "COORDINATE_NOISE_CONFIG.json"),
                "clean_cache_manifest_sha256": sha256_file(cache_root / "STRUCTURE_CACHE_MANIFEST.json"),
                "coordinate_cache_manifest_sha256": sha256_file(coordinate_cache_root / f"fold_{args.fold}" / f"seed_{args.seed}" / "COORDINATE_CACHE_MANIFEST.json")}
        histories["random2"], complete = train_branch_cpu("random2_coordinate", random2_state, train_atoms, train_lines, train_labels, train_ids,
            validation_atoms, validation_lines, validation_labels, validation_ids, args.seed, work_dir, integrity)
        if not complete: return 75
    if args.condition == "paired":
        control_order = [row["ordered_train_ids_sha256"] for row in histories["control"]]
        random2_order = [row["ordered_train_ids_sha256"] for row in histories["random2"]]
        if control_order != random2_order:
            raise RuntimeError("paired epoch sample ordering mismatch")
    (work_dir / "INCOMPLETE_RESUME_REQUIRED.json").unlink(missing_ok=True)
    write_json(work_dir / "TRAINING_STATUS.json", {"status": "training_complete", "fold": args.fold, "seed": args.seed,
        "condition": args.condition, "outer_test_accessed": False, "calibration_status": "blocked_until_validated_muben_ts"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

