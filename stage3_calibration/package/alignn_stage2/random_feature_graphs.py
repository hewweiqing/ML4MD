"""Stage A input set (c): real graph topology with node/edge/angle features
resampled from N(0,1).

Takes an already-built (atom_graph, line_graph) pair from real structures and
replaces only the floating-point feature tensors in ndata/edata with fresh
Gaussian draws, seeded deterministically. Topology (node count, edge count,
non-float index/id tensors) is left untouched and asserted unchanged, the
same invariant-checking discipline coordinate_noise.py uses for coordinate
perturbation.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import torch


def _tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def resample_features(graph, *, seed: int, namespaces: tuple[str, ...] = ("ndata", "edata")):
    """Return a deep copy of `graph` with every floating-point feature tensor
    in the given namespaces replaced by an independent N(0,1) draw of the
    same shape/dtype/device. Non-floating-point tensors (e.g. integer IDs)
    are left untouched. Raises if topology or any non-float tensor changed.
    """
    original_node_count = graph.num_nodes()
    original_edge_count = graph.num_edges()
    resampled = copy.deepcopy(graph)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    resampled_keys: list[str] = []
    for namespace_name in namespaces:
        namespace = getattr(resampled, namespace_name)
        for key in sorted(namespace.keys()):
            value = namespace[key]
            if not torch.is_tensor(value) or not torch.is_floating_point(value):
                continue
            noise = torch.randn(value.shape, generator=generator, dtype=torch.float32).to(
                dtype=value.dtype, device=value.device)
            if not torch.isfinite(noise).all():
                raise RuntimeError(f"non-finite resampled feature tensor: {namespace_name}.{key}")
            namespace[key] = noise
            resampled_keys.append(f"{namespace_name}.{key}")
    if resampled.num_nodes() != original_node_count or resampled.num_edges() != original_edge_count:
        raise RuntimeError("random-feature resampling changed graph topology")
    for namespace_name in namespaces:
        original_namespace, new_namespace = getattr(graph, namespace_name), getattr(resampled, namespace_name)
        for key in sorted(original_namespace.keys()):
            original_value = original_namespace[key]
            if torch.is_tensor(original_value) and torch.is_floating_point(original_value):
                continue
            if not torch.equal(original_value.cpu(), new_namespace[key].cpu()):
                raise RuntimeError(f"random-feature resampling changed a non-float tensor: {namespace_name}.{key}")
    return resampled, sorted(resampled_keys)


def derive_stream_seed(version: int, fold: int, seed: int, record_index: int, stream: str) -> int:
    """Same construction as coordinate_noise.derive_stream_seed, duplicated
    here (not imported) to keep this module usable standalone for Stage A's
    on-the-fly graph builder without importing coordinate_noise's PBC
    perturbation machinery.
    """
    payload = json.dumps([version, fold, seed, record_index, stream], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def build_random_feature_batch(atom_graphs: list, line_graphs: list, *, version: int, fold: int, seed: int):
    """Resample every (atom_graph, line_graph) pair in a batch, one
    independent seed per record via derive_stream_seed(..., "stage-a-random-feature").
    Returns (resampled_atom_graphs, resampled_line_graphs, provenance) where
    provenance is a list of per-record dicts: record index, seed used, the
    pre/post feature-tensor hashes, and which keys were resampled.
    """
    if len(atom_graphs) != len(line_graphs):
        raise RuntimeError("atom/line graph batch length mismatch")
    resampled_atoms, resampled_lines, provenance = [], [], []
    for index, (atom, line) in enumerate(zip(atom_graphs, line_graphs)):
        atom_seed = derive_stream_seed(version, fold, seed, index, "stage-a-random-feature-atom")
        line_seed = derive_stream_seed(version, fold, seed, index, "stage-a-random-feature-line")
        new_atom, atom_keys = resample_features(atom, seed=atom_seed)
        new_line, line_keys = resample_features(line, seed=line_seed)
        resampled_atoms.append(new_atom)
        resampled_lines.append(new_line)
        provenance.append({"record_index": index, "atom_seed": atom_seed, "line_seed": line_seed,
            "resampled_atom_keys": atom_keys, "resampled_line_keys": line_keys,
            "atom_node_count": new_atom.num_nodes(), "atom_edge_count": new_atom.num_edges(),
            "line_node_count": new_line.num_nodes(), "line_edge_count": new_line.num_edges()})
    return resampled_atoms, resampled_lines, provenance
