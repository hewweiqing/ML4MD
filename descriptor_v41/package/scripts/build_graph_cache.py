#!/usr/bin/env python3
"""Build/resume an atomically promoted, exhaustively verified structure cache."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import dgl
import ijson
from alignn.graphs import Graph
from jarvis.core.atoms import pmg_to_atoms
from pymatgen.core import Structure

from alignn_stage2.common import DATASET_SHA256, sha256_file, write_json
from alignn_stage2.structure_cache import (MANIFEST_NAME, SCHEMA_VERSION, SHARD_SIZE,
    atomic_write_shard, canonical_digest, expected_provenance, quarantine_shard,
    validate_manifest, verify_cache, verify_shard)
from alignn_stage2.training import GRAPH_SETTINGS


def release_identity(package_root: Path) -> str:
    manifest = json.loads((package_root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    value = manifest.get("aggregate_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError("package aggregate identity is unavailable")
    return value


def _quarantine_manifest(path: Path, quarantine_root: Path, reason: str) -> None:
    if not path.exists():
        return
    destination = Path(quarantine_root) / ("manifest_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_" + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    os.replace(path, destination / path.name)
    write_json(destination / "QUARANTINE_REASON.json", {"reason": reason,
        "quarantined_at_utc": datetime.now(timezone.utc).isoformat()})


def discover_shards(root: Path) -> set[int]:
    found = set()
    pattern = re.compile(r"^(?:atom|line|shard)_(\d{5})\.(?:bin|json)(?:\.tmp\..+)?$")
    for path in root.iterdir():
        match = pattern.match(path.name)
        if match and path.is_file():
            found.add(int(match.group(1)))
    return found


def existing_record(root: Path, shard: int, provenance_sha256: str) -> dict:
    meta_path = root / f"shard_{shard:05d}.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("provenance_sha256") != provenance_sha256:
        raise RuntimeError("metadata provenance mismatch")
    record = dict(metadata)
    record.update({"metadata_file": meta_path.name, "metadata_sha256": sha256_file(meta_path)})
    verify_shard(root, record, dgl_module=dgl, load_graphs=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--quarantine-root", required=True)
    parser.add_argument("--scope", choices=("full_label_free",), required=True)
    parser.add_argument("--package-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    dataset, root = Path(args.dataset), Path(args.cache_root)
    quarantine_root, package_root = Path(args.quarantine_root), Path(args.package_root)
    if sha256_file(dataset) != DATASET_SHA256:
        raise RuntimeError("official dataset hash mismatch")
    root.mkdir(parents=True, exist_ok=True)
    quarantine_root.mkdir(parents=True, exist_ok=True)
    if root.resolve() == quarantine_root.resolve():
        raise RuntimeError("cache and quarantine roots must differ")
    provenance = expected_provenance(GRAPH_SETTINGS, jarvis_version=version("jarvis-tools"),
        dgl_version=dgl.__version__, release_identity=release_identity(package_root), scope=args.scope)

    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_file():
        try:
            result = verify_cache(root, provenance, dgl_module=dgl, load_graphs=True)
            if result["verified_graphs"] != result["manifest"]["structure_count"]:
                raise RuntimeError("verified cache graph count mismatch")
            print("ALIGNN_GRAPH_CACHE: PASS (existing cache exhaustively verified)")
            return 0
        except Exception as error:
            _quarantine_manifest(manifest_path, quarantine_root, f"stale or invalid PASS manifest: {error}")
    for temporary_manifest in root.glob(f"{MANIFEST_NAME}.tmp.*"):
        _quarantine_manifest(temporary_manifest, quarantine_root, "interrupted manifest promotion")

    completed = {}
    for shard in sorted(discover_shards(root)):
        try:
            completed[shard] = existing_record(root, shard, provenance["provenance_sha256"])
        except Exception as error:
            quarantine_shard(root, quarantine_root, shard, f"incomplete/corrupt/stale shard: {error}")

    entries, shards = {}, []
    atom_buffer, line_buffer, id_buffer, hash_buffer = [], [], [], []
    active_shard = None

    def incorporate(record: dict) -> None:
        shard = int(record["shard"])
        if any(row["shard"] == shard for row in shards):
            return
        shards.append(record)
        for offset, (structure_id, structure_hash) in enumerate(zip(record["ids"], record["structure_sha256"])):
            if structure_id in entries:
                raise RuntimeError(f"duplicate structure ID while incorporating shard: {structure_id}")
            entries[structure_id] = {"shard": shard, "offset": offset, "structure_sha256": structure_hash}

    def flush(shard: int | None) -> None:
        nonlocal atom_buffer, line_buffer, id_buffer, hash_buffer
        if shard is None or not id_buffer:
            return
        # Any complete record was verified before streaming. A non-completed
        # shard is always rebuilt and atomically promoted, never overwritten in place.
        record = atomic_write_shard(root, shard, atom_buffer, line_buffer, id_buffer, hash_buffer,
            provenance_sha256=provenance["provenance_sha256"], dgl_module=dgl)
        verify_shard(root, record, dgl_module=dgl, load_graphs=True)
        incorporate(record)
        atom_buffer, line_buffer, id_buffer, hash_buffer = [], [], [], []

    structure_count = 0
    used_completed = set()
    with gzip.open(dataset, "rb") as stream:
        values = ijson.items(stream, "data.item.item", use_float=True)
        structures = (value for value in values if isinstance(value, dict))
        for global_index, structure_dict in enumerate(structures):
            structure_count += 1
            structure_id = f"mb-mp-is-metal-{global_index + 1:06d}"
            shard = global_index // SHARD_SIZE
            structure_hash = hashlib.sha256(json.dumps(structure_dict, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()
            if shard in completed:
                record = completed[shard]
                used_completed.add(shard)
                offset = global_index % SHARD_SIZE
                if offset >= len(record["ids"]) or record["ids"][offset] != structure_id \
                        or record["structure_sha256"][offset] != structure_hash:
                    quarantine_shard(root, quarantine_root, shard,
                        "dataset ID/order/structure hash disagrees with resumed shard")
                    completed.pop(shard)
                    raise RuntimeError(f"resumed shard {shard} dataset identity mismatch; rerun to rebuild quarantined evidence")
                incorporate(record)
                continue
            if active_shard is not None and shard != active_shard:
                flush(active_shard)
            active_shard = shard
            atoms = pmg_to_atoms(Structure.from_dict(structure_dict))
            atom, line = Graph.atom_dgl_multigraph(atoms=atoms, id=structure_id, **GRAPH_SETTINGS)
            atom_buffer.append(atom)
            line_buffer.append(line)
            id_buffer.append(structure_id)
            hash_buffer.append(structure_hash)
    flush(active_shard)
    for unused_shard in sorted(set(completed) - used_completed):
        quarantine_shard(root, quarantine_root, unused_shard, "valid shard is outside the current official dataset scope")
    shards.sort(key=lambda row: row["shard"])
    if structure_count != len(entries):
        raise RuntimeError(f"cache completeness mismatch: dataset={structure_count} entries={len(entries)}")
    manifest = {"schema_version": SCHEMA_VERSION, "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "release_identity": provenance["release_identity"],
        "provenance": provenance, "dataset_structure_count": structure_count,
        "structure_count": len(entries), "structure_ids_sha256": canonical_digest(sorted(entries)),
        "entries": entries, "shards": shards, "graph_constructions_this_invocation": structure_count - sum(
            completed[shard]["graph_count"] for shard in used_completed), "labels_used_in_graph_construction": False,
        "split_specific_graph_copies": False,
        "exhaustive_verification": {"status": "passed", "verified_shards": len(shards),
            "verified_graphs": len(entries), "method": "metadata, hashes, IDs, offsets, structure hashes, and DGL graph counts"}}
    validate_manifest(manifest, provenance)
    for record in shards:
        verify_shard(root, record, dgl_module=dgl, load_graphs=True)
    temporary = root / f"{MANIFEST_NAME}.tmp.{uuid.uuid4().hex}"
    write_json(temporary, manifest)
    os.replace(temporary, manifest_path)
    # Re-read and exhaustively verify the promoted PASS evidence.
    result = verify_cache(root, provenance, dgl_module=dgl, load_graphs=True)
    if result["verified_graphs"] != structure_count:
        raise RuntimeError("post-promotion exhaustive cache verification failed")
    print("ALIGNN_GRAPH_CACHE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
