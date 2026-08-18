"""Outcome-neutral Cartesian perturbation with axis-selective PBC wrapping."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PerturbationRecord:
    displacement: np.ndarray
    wrapped_counts: tuple[int, int, int]
    clean_coordinate_sha256: str
    displacement_sha256: str
    perturbed_coordinate_sha256: str


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def perturb_structure(structure, sigma: float, seed: int):
    """Deep-copy a pymatgen Structure; never reads or uses a target label."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    clean = np.asarray(structure.cart_coords, dtype=np.float64).copy()
    generator = np.random.default_rng(int(seed))
    displacement = generator.normal(0.0, float(sigma), size=clean.shape)
    if not np.isfinite(displacement).all():
        raise RuntimeError("non-finite displacement")
    noisy_cart = clean + displacement
    lattice = copy.deepcopy(structure.lattice)
    fractional = np.asarray(lattice.get_fractional_coords(noisy_cart), dtype=np.float64)
    pbc = tuple(bool(x) for x in getattr(lattice, "pbc", (True, True, True)))
    wrapped = [0, 0, 0]
    for axis, periodic in enumerate(pbc):
        if periodic:
            before = fractional[:, axis].copy()
            fractional[:, axis] = np.mod(fractional[:, axis], 1.0)
            wrapped[axis] = int(np.count_nonzero(~np.isclose(before, fractional[:, axis])))
    noisy = structure.__class__(lattice, list(structure.species), fractional, coords_are_cartesian=False,
        site_properties=copy.deepcopy(structure.site_properties), charge=getattr(structure, "charge", None))
    if list(noisy.species) != list(structure.species) or len(noisy) != len(structure):
        raise RuntimeError("species or structure length changed")
    if not np.array_equal(np.asarray(noisy.lattice.matrix), np.asarray(structure.lattice.matrix)):
        raise RuntimeError("unit cell changed")
    if tuple(bool(x) for x in noisy.lattice.pbc) != pbc:
        raise RuntimeError("PBC flags changed")
    if not np.array_equal(np.asarray(structure.cart_coords), clean):
        raise RuntimeError("original structure mutated")
    record = PerturbationRecord(displacement, tuple(wrapped), array_sha256(clean),
        array_sha256(displacement), array_sha256(np.asarray(noisy.cart_coords)))
    return noisy, record


def derive_stream_seed(version: int, fold: int, seed: int, record_index: int, stream: str) -> int:
    payload = json.dumps([version, fold, seed, record_index, stream], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def balanced_random_labels(count: int, seed: int) -> np.ndarray:
    if count <= 0 or count % 2:
        raise ValueError("balanced-label count must be positive and even")
    labels = np.asarray([0] * (count // 2) + [1] * (count // 2), dtype=np.int64)
    np.random.default_rng(seed).shuffle(labels)
    return labels

