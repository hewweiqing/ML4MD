"""Stage A initialization diagnostic orchestration.

Requires alignn/dgl/CUDA (see STAGE3_STAGE_A_PROTOCOL.md) — cannot be run on
a machine without that stack. This module is written and structurally
reviewed, but its actual measurements have not been executed anywhere; the
first real run must also dump ALIGNN_MODULE_INVENTORY.json (see
scripts/run_stage_a_diagnostic.py) before the labeled activation trace can
be trusted for semantic module names.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from .activation_probe import (activation_scale_summary, apply_head_perturbation_control,
    evaluate_mechanical_gate, head_displacement_norm, logit_diagnostics, mechanism_verdict,
    register_activation_probe, remove_hooks)
from .common import STAGE3_VERSION, sha256_ids, write_json
from .coordinate_head_warmup import extract_pooled_descriptors, warm_coordinate_head
from .coordinate_noise import balanced_random_labels, perturb_structure
from .coordinate_noise import derive_stream_seed as coordinate_derive_stream_seed
from .cpu_split import GRAPH_SETTINGS, load_inner_split
from .descriptor_head_warmup import warm_descriptor_head
from .full_network_warmup import warm_full_network
from .gpu_runtime import DEVICE, assert_model_gpu, configure_gpu_runtime
from .random_feature_graphs import build_random_feature_batch, derive_stream_seed
from .sigma_authorization import require_authorized_sigma

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
STAGE_A_SEEDS = tuple(range(20))
MIN_STRUCTURES = 2000
GRAPH_BATCH_SIZE = 32
COORDINATE_RECORD_COUNT = 3000
INPUT_SETS = ("real", "coordinate_perturbed", "random_feature")
WARMUP_VARIANTS = ("descriptor", "coordinate", "full_network")


def model_config():
    from alignn.models.alignn import ALIGNNConfig
    return ALIGNNConfig(name="alignn", alignn_layers=4, gcn_layers=4, atom_input_features=92,
        edge_input_features=80, triplet_input_features=40, embedding_features=64,
        hidden_features=256, classification=True, num_classes=2, link="identity")


def build_model():
    from alignn.models.alignn import ALIGNN
    return ALIGNN(model_config()).to(DEVICE)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _stable_order(ids: list[str], fold: int, seed: int) -> list[str]:
    return sorted(ids, key=lambda item: hashlib.sha256(
        f"stage3|v{STAGE3_VERSION}|source|{fold}|{seed}|{item}".encode()).digest())


def _load_structures(dataset: Path, selected: set[str]) -> dict:
    from pymatgen.core import Structure
    import ijson
    output = {}
    with gzip.open(dataset, "rb") as stream:
        structures = (value for value in ijson.items(stream, "data.item.item", use_float=True) if isinstance(value, dict))
        for index, value in enumerate(structures):
            identity = f"mb-mp-is-metal-{index + 1:06d}"
            if identity in selected:
                output[identity] = Structure.from_dict(value)
    if set(output) != selected:
        raise RuntimeError("selected Stage A structure coverage mismatch")
    return output


def _build_graphs(structures: list, ids: list[str], id_prefix: str) -> tuple[list, list]:
    from alignn.graphs import Graph
    from jarvis.core.atoms import pmg_to_atoms
    atom_graphs, line_graphs = [], []
    for structure_id, structure in zip(ids, structures):
        atom, line = Graph.atom_dgl_multigraph(atoms=pmg_to_atoms(structure), id=f"{id_prefix}-{structure_id}", **GRAPH_SETTINGS)
        atom_graphs.append(atom)
        line_graphs.append(line)
    return atom_graphs, line_graphs


def build_real_structure_input(dataset: Path, fold: int, seed: int, count: int = MIN_STRUCTURES):
    train_ids, _, _, _ = load_inner_split(dataset, fold, seed)
    ordered = _stable_order(train_ids, fold, seed)[:count]
    structures_by_id = _load_structures(dataset, set(ordered))
    structures = [structures_by_id[item] for item in ordered]
    atom_graphs, line_graphs = _build_graphs(structures, ordered, "stage3-a-real")
    return {"ids": ordered, "ids_sha256": sha256_ids(ordered), "atom_graphs": atom_graphs,
        "line_graphs": line_graphs, "structures": structures}


def build_coordinate_perturbed_input(real_input: dict, *, sigma: float, fold: int, seed: int):
    perturbed_structures = []
    for index, (structure_id, structure) in enumerate(zip(real_input["ids"], real_input["structures"])):
        displacement_seed = coordinate_derive_stream_seed(STAGE3_VERSION, fold, seed, index, "stage3-a-coordinate")
        noisy, _record = perturb_structure(structure, sigma, displacement_seed)
        perturbed_structures.append(noisy)
    atom_graphs, line_graphs = _build_graphs(perturbed_structures, real_input["ids"], "stage3-a-coordinate")
    return {"ids": real_input["ids"], "atom_graphs": atom_graphs, "line_graphs": line_graphs}


def build_random_feature_input(real_input: dict, *, fold: int, seed: int):
    atom_graphs, line_graphs, provenance = build_random_feature_batch(
        real_input["atom_graphs"], real_input["line_graphs"], version=STAGE3_VERSION, fold=fold, seed=seed)
    return {"ids": real_input["ids"], "atom_graphs": atom_graphs, "line_graphs": line_graphs, "provenance": provenance}


def graph_batch(atom_graphs: list, line_graphs: list, indices: list[int]):
    import dgl
    atoms = dgl.batch([atom_graphs[i] for i in indices]).to(DEVICE)
    lines = dgl.batch([line_graphs[i] for i in indices]).to(DEVICE)
    return atoms, lines


def native_logits(model: torch.nn.Module, batch_input) -> torch.Tensor:
    captured = []
    hook = model.fc.register_forward_hook(lambda _module, _inputs, output: captured.append(output))
    model(batch_input)
    hook.remove()
    return captured[-1]


def measure(model: torch.nn.Module, input_set: dict) -> dict:
    """One forward pass in eval/no_grad over an entire input set, with the
    activation probe attached, plus logit diagnostics on the native logits.
    """
    model.eval()
    trace, handles = register_activation_probe(model)
    logits_batches = []
    with torch.no_grad():
        for start in range(0, len(input_set["atom_graphs"]), GRAPH_BATCH_SIZE):
            indices = list(range(start, min(start + GRAPH_BATCH_SIZE, len(input_set["atom_graphs"]))))
            batch_input = graph_batch(input_set["atom_graphs"], input_set["line_graphs"], indices)
            logits_batches.append(native_logits(model, batch_input))
    remove_hooks(handles)
    logits = torch.cat(logits_batches)
    diagnostics = logit_diagnostics(logits)
    mean_positive_probability = float(torch.softmax(logits, dim=1)[:, 1].mean())
    return {"logit_diagnostics": diagnostics, "mechanism_verdict": mechanism_verdict(diagnostics),
        "mechanical_gate": evaluate_mechanical_gate(mean_positive_probability, diagnostics["predictive_entropy_mean"]),
        "activation_scale_summary": activation_scale_summary(trace)}


def make_full_network_batch_factory(real_input: dict, *, fold: int):
    """Pure function of (step, seed): fresh noise features + fresh balanced
    random labels every call, both on DEVICE, reproducible given the same
    (step, seed) pair (required for warm_full_network's deterministic-replay
    check).
    """
    def factory(step: int, replay_seed: int):
        step_seed = derive_stream_seed(STAGE3_VERSION, fold, replay_seed, step, "stage3-a-full-network-step")
        atoms, lines, _ = build_random_feature_batch(real_input["atom_graphs"][:128], real_input["line_graphs"][:128],
            version=STAGE3_VERSION, fold=fold, seed=step_seed)
        labels = torch.as_tensor(balanced_random_labels(128, step_seed + 1), dtype=torch.long, device=DEVICE)
        return graph_batch(atoms, lines, list(range(len(atoms)))), labels
    return factory


def time_full_network_warmup_single_seed(dataset: Path, fold: int, seed: int) -> dict:
    """Runs ONLY the full-network warm-up (not the rest of Stage A) for one
    (fold, seed) and reports wall-clock time. This is the "time one seed of
    the full-network arm first" profiling step requested before submitting
    the full 20-seed Stage A run — 938 whole-network optimizer steps run
    TWICE (deterministic-replay contract) is a much larger intervention than
    the head-only arms, so its cost should be measured before committing to
    the full grid, not assumed.
    """
    import json
    import time
    configure_gpu_runtime()
    seed_all(seed)
    real_input = build_real_structure_input(dataset, fold, seed)
    initial_state = {name: value.detach().clone() for name, value in build_model().state_dict().items()}
    started = time.perf_counter()
    result = warm_full_network(build_model, initial_state, make_full_network_batch_factory(real_input, fold=fold),
        native_logits, seed=seed)
    elapsed_seconds = time.perf_counter() - started
    manifest = json.loads((PACKAGE_ROOT / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    return {"status": "passed", "fold": fold, "seed": seed, "elapsed_seconds": elapsed_seconds,
        "elapsed_seconds_per_replay": elapsed_seconds / 2, "optimizer_steps_per_replay": result["optimizer_steps"],
        "batch_size": result["batch_size"], "deterministic_replay": result["deterministic_replay"],
        "projected_seconds_for_20_seeds": elapsed_seconds * len(STAGE_A_SEEDS),
        "package_aggregate_sha256": manifest["aggregate_sha256"], "diagnostics": result["diagnostics"]}


def cross_seed_verdict_summary(results: list[dict]) -> dict:
    """Per (phase, input_set), how the mechanism_verdict is distributed
    across seeds. If a verdict splits across seeds rather than being
    consistent, that itself is a finding (initialization variance large
    enough to matter, per review feedback) — this makes that visible
    without hand-inspecting 20 seeds' worth of nested JSON.
    """
    buckets: dict[tuple[str, str], dict[str, int]] = {}
    for row in results:
        for phase_name, phase in (("pre_warmup", row["pre_warmup"]), *row["post_warmup"].items()):
            if not isinstance(phase, dict):
                continue
            for input_set_name in INPUT_SETS:
                entry = phase.get(input_set_name)
                if not isinstance(entry, dict) or "mechanism_verdict" not in entry:
                    continue
                key = (phase_name, input_set_name)
                buckets.setdefault(key, {})
                verdict = entry["mechanism_verdict"]
                buckets[key][verdict] = buckets[key].get(verdict, 0) + 1
    return {f"{phase}::{input_set}": {"counts": counts, "consistent_across_seeds": len(counts) == 1}
        for (phase, input_set), counts in sorted(buckets.items())}


def run_stage_a(dataset: Path, fold: int, work_dir: Path, *, sigma_authorization_path: str | None = None) -> dict:
    configure_gpu_runtime()
    authorization = require_authorized_sigma(authorization_path=sigma_authorization_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in STAGE_A_SEEDS:
        seed_all(seed)
        real_input = build_real_structure_input(dataset, fold, seed)
        coordinate_input = build_coordinate_perturbed_input(real_input, sigma=authorization.sigma, fold=fold, seed=seed)
        random_feature_input = build_random_feature_input(real_input, fold=fold, seed=seed)
        input_sets = {"real": real_input, "coordinate_perturbed": coordinate_input, "random_feature": random_feature_input}

        model = build_model()
        assert_model_gpu(model)
        pre_warmup = {name: measure(model, input_sets[name]) for name in INPUT_SETS}

        initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        descriptors = extract_pooled_descriptors(model, real_input["atom_graphs"], real_input["line_graphs"], graph_batch)
        descriptor_warmup = warm_descriptor_head(model.fc.state_dict(), descriptors, seed=seed)

        coordinate_descriptors = extract_pooled_descriptors(model, coordinate_input["atom_graphs"], coordinate_input["line_graphs"], graph_batch)
        coordinate_labels = torch.as_tensor(balanced_random_labels(COORDINATE_RECORD_COUNT,
            coordinate_derive_stream_seed(STAGE3_VERSION, fold, seed, 0, "stage3-a-coordinate-labels")), dtype=torch.long, device=DEVICE)
        coordinate_warmup = warm_coordinate_head(model.fc.state_dict(), coordinate_descriptors[:COORDINATE_RECORD_COUNT],
            coordinate_labels, version=STAGE3_VERSION, fold=fold, seed=seed)

        full_network_result = warm_full_network(build_model, initial_state,
            make_full_network_batch_factory(real_input, fold=fold), native_logits, seed=seed)

        post_warmup = {}
        for variant, fc_state in (("descriptor", descriptor_warmup["fc_state"]),
                ("coordinate", coordinate_warmup["fc_state"])):
            warmed = build_model()
            warmed.load_state_dict(initial_state)
            warmed.fc.load_state_dict(fc_state)
            post_warmup[variant] = {name: measure(warmed, input_sets[name]) for name in INPUT_SETS}
        full_network_model = build_model()
        full_network_model.load_state_dict(full_network_result["final_state"])
        post_warmup["full_network"] = {name: measure(full_network_model, input_sets[name]) for name in INPUT_SETS}

        control_norm = head_displacement_norm(
            initial_state["fc.weight"], initial_state["fc.bias"], descriptor_warmup["fc_state"]["weight"], descriptor_warmup["fc_state"]["bias"])
        control_model = build_model()
        control_model.load_state_dict(initial_state)
        control_provenance = apply_head_perturbation_control(control_model, target_norm=control_norm, seed=seed + 900000)
        post_warmup["head_perturbation_control"] = {name: measure(control_model, input_sets[name]) for name in INPUT_SETS}
        post_warmup["head_perturbation_control"]["provenance"] = control_provenance

        results.append({"seed": seed, "fold": fold, "input_set_ids_sha256": real_input["ids_sha256"],
            "pre_warmup": pre_warmup, "post_warmup": post_warmup,
            "descriptor_warmup_diagnostics": descriptor_warmup["diagnostics"],
            "coordinate_warmup_diagnostics": coordinate_warmup["diagnostics"],
            "full_network_warmup_diagnostics": full_network_result["diagnostics"],
            "authorization_sha256": authorization.authorization_sha256,
            "sigma_cartesian_per_axis_angstrom": authorization.sigma})

    report = {"status": "passed", "stage": "A", "fold": fold, "seeds": list(STAGE_A_SEEDS),
        "input_sets": list(INPUT_SETS), "warmup_variants": list(WARMUP_VARIANTS) + ["head_perturbation_control"],
        "cross_seed_verdict_summary": cross_seed_verdict_summary(results),
        "results": results}
    write_json(work_dir / f"STAGE_A_DIAGNOSTIC_fold{fold}.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--sigma-authorization", default=None)
    args = parser.parse_args(argv)
    run_stage_a(Path(args.dataset), args.fold, Path(args.work_dir), sigma_authorization_path=args.sigma_authorization)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
