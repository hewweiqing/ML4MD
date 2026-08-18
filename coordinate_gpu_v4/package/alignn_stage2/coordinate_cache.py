"""Fold/seed coordinate warm-up cache with atomic shards and exhaustive hashes."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import uuid
from importlib.metadata import version
from pathlib import Path

import dgl
import ijson
import numpy as np
import torch
from alignn.graphs import Graph
from jarvis.core.atoms import pmg_to_atoms
from pymatgen.core import Structure

from .common import DATASET_SHA256, sha256_file, sha256_ids, write_json
from .coordinate_noise import balanced_random_labels, derive_stream_seed, perturb_structure
from .sigma_authorization import AuthorizedSigma, require_authorized_sigma
from .cpu_split import GRAPH_SETTINGS, load_inner_split


MANIFEST = "COORDINATE_CACHE_MANIFEST.json"
SHARD_SIZE = 256
RECORD_COUNT = 3000
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def graph_digest(graph) -> str:
    digest = hashlib.sha256()
    src, dst = graph.edges(order="eid")
    digest.update(np.asarray(src.cpu(), dtype=np.int64).tobytes())
    digest.update(np.asarray(dst.cpu(), dtype=np.int64).tobytes())
    for namespace in (graph.ndata, graph.edata):
        for key in sorted(namespace):
            digest.update(key.encode()); digest.update(namespace[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def graph_feature_tensors(graph):
    """Materialize DGL features by key; DGL 1.1.1 `.values()` yields lazy Columns."""
    return [namespace[key] for namespace in (graph.ndata, graph.edata) for key in sorted(namespace.keys())]


def validate_graph(graph, name: str) -> None:
    if graph.num_nodes() <= 0:
        raise RuntimeError(f"{name} has no nodes")
    tensors = graph_feature_tensors(graph)
    if not tensors:
        raise RuntimeError(f"{name} has no node/edge feature tensors")
    if not all(isinstance(value, torch.Tensor) and torch.isfinite(value).all().item() for value in tensors):
        raise RuntimeError(f"{name} contains a non-tensor or non-finite feature")


def _load_structures(dataset: Path, selected: set[str]) -> dict[str, Structure]:
    output = {}
    with gzip.open(dataset, "rb") as stream:
        structures = (value for value in ijson.items(stream, "data.item.item", use_float=True) if isinstance(value, dict))
        for index, value in enumerate(structures):
            identity = f"mb-mp-is-metal-{index + 1:06d}"
            if identity in selected:
                output[identity] = Structure.from_dict(value)
    if set(output) != selected:
        raise RuntimeError("selected inner-training structure coverage mismatch")
    return output


def _stable_order(ids: list[str], fold: int, seed: int) -> list[str]:
    return sorted(ids, key=lambda item: hashlib.sha256(f"v28|source|{fold}|{seed}|{item}".encode()).digest())


def _write_shard(root: Path, shard: int, atoms: list, lines: list, records: list[dict]) -> dict:
    token = uuid.uuid4().hex
    names = {"atom": f"atom_{shard:05d}.bin", "line": f"line_{shard:05d}.bin", "record": f"records_{shard:05d}.json"}
    temporary = {key: root / f"{name}.tmp.{token}" for key, name in names.items()}
    dgl.save_graphs(str(temporary["atom"]), atoms); dgl.save_graphs(str(temporary["line"]), lines)
    temporary["record"].write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {"shard": shard, "count": len(records)}
    for key, name in names.items():
        result[f"{key}_file"] = name; result[f"{key}_sha256"] = sha256_file(temporary[key])
    for key, name in names.items():
        os.replace(temporary[key], root / name)
    return result


def verify_coordinate_cache(root: str | Path, *, expected_fold: int | None = None,
        expected_seed: int | None = None, authorization: AuthorizedSigma | None = None) -> dict:
    root = Path(root); path = root / MANIFEST
    if not path.is_file():
        raise RuntimeError("coordinate cache manifest missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "passed" or value.get("record_count") != RECORD_COUNT:
        raise RuntimeError("coordinate cache is incomplete")
    if expected_fold is not None and value.get("fold") != expected_fold or expected_seed is not None and value.get("seed") != expected_seed:
        raise RuntimeError("coordinate cache cell mismatch")
    if authorization and (value.get("authorization_sha256") != authorization.authorization_sha256
            or value.get("sigma_cartesian_per_axis_angstrom") != authorization.sigma):
        raise RuntimeError("coordinate cache authorization mismatch")
    count = 0
    for row in value["shards"]:
        for key in ("atom", "line", "record"):
            file = root / row[f"{key}_file"]
            if not file.is_file() or sha256_file(file) != row[f"{key}_sha256"]:
                raise RuntimeError(f"coordinate shard hash mismatch: {file.name}")
        atoms = dgl.load_graphs(str(root / row["atom_file"]))[0]
        lines = dgl.load_graphs(str(root / row["line_file"]))[0]
        records = json.loads((root / row["record_file"]).read_text(encoding="utf-8"))
        if len(atoms) != row["count"] or len(lines) != row["count"] or len(records) != row["count"]:
            raise RuntimeError("partial coordinate shard cannot be certified")
        count += row["count"]
    if count != RECORD_COUNT:
        raise RuntimeError("coordinate cache exhaustive count mismatch")
    return {"status": "passed", "verified_records": count, "manifest_sha256": sha256_file(path), "manifest": value}


def load_coordinate_graphs(root: str | Path, authorization: AuthorizedSigma):
    result = verify_coordinate_cache(root, authorization=authorization); root = Path(root)
    atoms, lines, records = [], [], []
    for row in result["manifest"]["shards"]:
        atoms.extend(dgl.load_graphs(str(root / row["atom_file"]))[0])
        lines.extend(dgl.load_graphs(str(root / row["line_file"]))[0])
        records.extend(json.loads((root / row["record_file"]).read_text(encoding="utf-8")))
    return atoms, lines, records


def build_coordinate_cache(dataset: str | Path, root: str | Path, fold: int, seed: int,
        *, authorization_path: str | Path | None = None) -> dict:
    authorization = require_authorized_sigma(authorization_path=authorization_path)
    dataset, root = Path(dataset), Path(root)
    if sha256_file(dataset) != DATASET_SHA256:
        raise RuntimeError("official dataset hash mismatch")
    cell = root / f"fold_{fold}" / f"seed_{seed}"; cell.mkdir(parents=True, exist_ok=True)
    if (cell / MANIFEST).is_file():
        return verify_coordinate_cache(cell, expected_fold=fold, expected_seed=seed, authorization=authorization)
    train_ids, validation_ids, _train_labels, _validation_labels = load_inner_split(dataset, fold, seed)
    ordered = _stable_order(train_ids, fold, seed)
    if len(ordered) < RECORD_COUNT:
        raise RuntimeError("fewer than 3000 inner-training structures")
    candidates = ordered[:RECORD_COUNT + 1000]
    structures = _load_structures(dataset, set(candidates))
    labels = balanced_random_labels(RECORD_COUNT, derive_stream_seed(28, fold, seed, 0, "balanced-label-order"))
    atom_buffer, line_buffer, record_buffer, shards, accepted, rejected = [], [], [], [], 0, []
    for occurrence, base_id in enumerate(candidates):
        if accepted == RECORD_COUNT: break
        displacement_seed = derive_stream_seed(28, fold, seed, occurrence, "coordinate-displacement")
        try:
            clean = structures[base_id]
            noisy, perturbation = perturb_structure(clean, authorization.sigma, displacement_seed)
            atom, line = Graph.atom_dgl_multigraph(atoms=pmg_to_atoms(noisy),
                id=f"v28-f{fold}-s{seed}-r{accepted:04d}", **GRAPH_SETTINGS)
            validate_graph(atom, "atom_graph")
            validate_graph(line, "line_graph")
            clean_hash = canonical(clean.as_dict()); noisy_hash = canonical(noisy.as_dict())
            record = {"warmup_record_id": f"v28-f{fold}-s{seed}-r{accepted:04d}", "base_structure_id": base_id,
                "fold": fold, "experiment_seed": seed, "occurrence_index": occurrence,
                "base_structure_sha256": clean_hash, "clean_coordinate_sha256": perturbation.clean_coordinate_sha256,
                "displacement_array_sha256": perturbation.displacement_sha256,
                "perturbed_coordinate_sha256": perturbation.perturbed_coordinate_sha256,
                "noisy_structure_sha256": noisy_hash, "atom_graph_sha256": graph_digest(atom),
                "line_graph_sha256": graph_digest(line), "species_sha256": canonical([str(x) for x in clean.species]),
                "lattice_sha256": canonical(np.asarray(clean.lattice.matrix).tolist()),
                "pbc_flags": list(map(bool, clean.lattice.pbc)), "displacement_seed": displacement_seed,
                "displacement_summary": {"mean": float(perturbation.displacement.mean()),
                    "std": float(perturbation.displacement.std()), "maximum_absolute": float(np.abs(perturbation.displacement).max()),
                    "mean_norm": float(np.linalg.norm(perturbation.displacement, axis=1).mean())},
                "minimum_interatomic_distance": float(noisy.distance_matrix[np.triu_indices(len(noisy), 1)].min()) if len(noisy) > 1 else None,
                "atom_edge_count": int(atom.num_edges()), "line_graph_edge_count": int(line.num_edges()),
                "neighbor_count_summary": {"mean": float(atom.in_degrees().float().mean()), "max": int(atom.in_degrees().max())},
                "wrapped_counts_by_direction": list(perturbation.wrapped_counts), "validity_status": "passed",
                "random_label": int(labels[accepted]), "true_label_used": False,
                "cache_key_sha256": canonical([28, fold, seed, occurrence, base_id, authorization.authorization_sha256,
                    sha256_file(PACKAGE_ROOT / "COORDINATE_NOISE_CONFIG.json"), GRAPH_SETTINGS])}
            atom_buffer.append(atom); line_buffer.append(line); record_buffer.append(record); accepted += 1
            if len(record_buffer) == SHARD_SIZE:
                shards.append(_write_shard(cell, len(shards), atom_buffer, line_buffer, record_buffer))
                atom_buffer, line_buffer, record_buffer = [], [], []
        except Exception as error:
            rejected.append({"base_structure_id": base_id, "occurrence_index": occurrence, "error": str(error)})
    if accepted != RECORD_COUNT:
        diagnostic = {"status": "failed_outcome_neutral_coordinate_cache_validation", "fold": fold, "seed": seed,
            "accepted": accepted, "required": RECORD_COUNT, "candidate_count": len(candidates),
            "authorization_sha256": authorization.authorization_sha256,
            "sigma_cartesian_per_axis_angstrom": authorization.sigma,
            "rejection_count": len(rejected), "first_rejections": rejected[:25]}
        temporary_diagnostic = cell / f"COORDINATE_CACHE_FAILURE_DIAGNOSTIC.json.tmp.{uuid.uuid4().hex}"
        write_json(temporary_diagnostic, diagnostic)
        os.replace(temporary_diagnostic, cell / "COORDINATE_CACHE_FAILURE_DIAGNOSTIC.json")
        first = rejected[0]["error"] if rejected else "no rejection exception recorded"
        raise RuntimeError(f"unable to obtain exactly 3000 valid records within frozen resampling limit; "
            f"accepted={accepted}; first_rejection={first}; diagnostic={cell / 'COORDINATE_CACHE_FAILURE_DIAGNOSTIC.json'}")
    if record_buffer:
        shards.append(_write_shard(cell, len(shards), atom_buffer, line_buffer, record_buffer))
    package = json.loads((PACKAGE_ROOT / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    manifest = {"schema_version": 1, "status": "passed", "package_release": 29,
        "package_aggregate_sha256": package["aggregate_sha256"], "fold": fold, "seed": seed,
        "record_count": accepted, "warmup_ids_sha256": sha256_ids(r["warmup_record_id"] for r in sum(
            [json.loads((cell / s["record_file"]).read_text(encoding="utf-8")) for s in shards], [])),
        "inner_train_ids_sha256": sha256_ids(train_ids), "inner_validation_ids_sha256": sha256_ids(validation_ids),
        "inner_validation_intersection_count": 0, "outer_test_structures_used": False, "true_labels_used": False,
        "sigma_cartesian_per_axis_angstrom": authorization.sigma,
        "authorization_sha256": authorization.authorization_sha256,
        "coordinate_noise_config_sha256": sha256_file(PACKAGE_ROOT / "COORDINATE_NOISE_CONFIG.json"),
        "graph_builder_sha256": sha256_file(Path(__file__)), "graph_settings": GRAPH_SETTINGS,
        "dependencies": {"dgl": dgl.__version__, "jarvis-tools": version("jarvis-tools")},
        "shards": shards, "rejected_and_resampled": rejected, "exhaustive_verification": "pending_post_promotion"}
    temporary = cell / f"{MANIFEST}.tmp.{uuid.uuid4().hex}"
    write_json(temporary, manifest); os.replace(temporary, cell / MANIFEST)
    manifest["exhaustive_verification"] = verify_coordinate_cache(cell, expected_fold=fold, expected_seed=seed,
        authorization=authorization)["status"]
    write_json(cell / MANIFEST, manifest)
    return verify_coordinate_cache(cell, expected_fold=fold, expected_seed=seed, authorization=authorization)
