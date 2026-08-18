"""Verified structure-level cache with complete provenance and safe recovery."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .common import ALIGNN_COMMIT, DATASET_SHA256, sha256_file

SHARD_SIZE = 512
MANIFEST_NAME = "STRUCTURE_CACHE_MANIFEST.json"
SCHEMA_VERSION = 2
_SHARD_MEMORY_CACHE = {}


def canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def graph_settings_digest(settings: dict) -> str:
    return canonical_digest(settings)


def expected_provenance(settings: dict, *, jarvis_version: str, dgl_version: str,
        release_identity: str, dataset_sha256: str = DATASET_SHA256,
        alignn_commit: str = ALIGNN_COMMIT, scope: str = "full_label_free") -> dict:
    value = {
        "dataset_sha256": dataset_sha256,
        "graph_settings": settings,
        "graph_settings_sha256": graph_settings_digest(settings),
        "alignn_commit": alignn_commit,
        "jarvis_tools_version": jarvis_version,
        "dgl_version": dgl_version,
        "shard_size": SHARD_SIZE,
        "scope": scope,
        "label_free_policy": "OUTER_TEST_STRUCTURE_PREPROCESSING_POLICY.md",
        "labels_used_in_graph_construction": False,
        "release_identity": release_identity,
    }
    value["provenance_sha256"] = canonical_digest(value)
    return value


def settings_digest(settings: dict, *, alignn_commit: str, jarvis_version: str, dgl_version: str) -> str:
    """Compatibility helper retained for callers; v11 uses complete provenance."""
    return expected_provenance(settings, alignn_commit=alignn_commit, jarvis_version=jarvis_version,
        dgl_version=dgl_version, release_identity="compatibility-only")["provenance_sha256"]


def shard_number(global_index: int) -> int:
    if global_index < 0:
        raise ValueError("global index must be non-negative")
    return global_index // SHARD_SIZE


def _require_hex_digest(value: object, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise RuntimeError(f"invalid {name}")


def validate_manifest(manifest: dict, expected: dict | None = None) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "passed":
        raise RuntimeError("structure-cache manifest is not a schema-v2 PASS manifest")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("structure-cache provenance is missing")
    if provenance.get("provenance_sha256") != canonical_digest({k: v for k, v in provenance.items() if k != "provenance_sha256"}):
        raise RuntimeError("structure-cache provenance digest mismatch")
    if expected is not None and provenance != expected:
        raise RuntimeError("structure-cache provenance does not match the current dataset/settings/runtime/release")
    entries, shards = manifest.get("entries"), manifest.get("shards")
    if not isinstance(entries, dict) or not entries or not isinstance(shards, list) or not shards:
        raise RuntimeError("invalid or empty structure-cache manifest")
    if manifest.get("structure_count") != len(entries):
        raise RuntimeError("structure-cache count mismatch")
    if manifest.get("dataset_structure_count") != len(entries):
        raise RuntimeError("dataset/cache structure-count mismatch")
    if manifest.get("release_identity") != provenance.get("release_identity"):
        raise RuntimeError("cache manifest release identity mismatch")
    if manifest.get("structure_ids_sha256") != canonical_digest(sorted(entries)):
        raise RuntimeError("structure-ID digest mismatch")
    locations = []
    shard_numbers = set()
    for record in shards:
        shard = record.get("shard")
        if not isinstance(shard, int) or shard in shard_numbers:
            raise RuntimeError("duplicate or invalid shard number")
        shard_numbers.add(shard)
        ids, offsets, hashes = record.get("ids"), record.get("offsets"), record.get("structure_sha256")
        if not isinstance(ids, list) or not ids or offsets != list(range(len(ids))) or len(hashes or []) != len(ids):
            raise RuntimeError(f"shard {shard} ID/offset/structure-hash metadata mismatch")
        if record.get("graph_count") != len(ids) or len(ids) > provenance["shard_size"]:
            raise RuntimeError(f"shard {shard} graph count mismatch")
        for key in ("atom_sha256", "line_sha256", "metadata_sha256"):
            _require_hex_digest(record.get(key), f"shard {shard} {key}")
        for offset, (structure_id, structure_hash) in enumerate(zip(ids, hashes)):
            _require_hex_digest(structure_hash, f"structure {structure_id} hash")
            expected_entry = {"shard": shard, "offset": offset, "structure_sha256": structure_hash}
            if entries.get(structure_id) != expected_entry:
                raise RuntimeError(f"entry mismatch for {structure_id}")
            locations.append((shard, offset))
    if len(locations) != len(set(locations)) or len(locations) != len(entries):
        raise RuntimeError("duplicate shard offsets or unreferenced entries")
    verification = manifest.get("exhaustive_verification", {})
    if verification.get("status") != "passed" or verification.get("verified_shards") != len(shards):
        raise RuntimeError("cache has no complete exhaustive PASS verification")


def verify_shard(cache_root: Path, record: dict, *, dgl_module, load_graphs: bool = True) -> dict:
    cache_root = Path(cache_root)
    shard = int(record["shard"])
    atom_path = cache_root / record["atom_file"]
    line_path = cache_root / record["line_file"]
    meta_path = cache_root / record["metadata_file"]
    for path in (atom_path, line_path, meta_path):
        if not path.is_file():
            raise RuntimeError(f"shard {shard} file missing: {path.name}")
    if sha256_file(atom_path) != record["atom_sha256"] or sha256_file(line_path) != record["line_sha256"]:
        raise RuntimeError(f"structure-cache shard {shard} graph hash mismatch")
    if sha256_file(meta_path) != record["metadata_sha256"]:
        raise RuntimeError(f"structure-cache shard {shard} metadata hash mismatch")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ("shard", "ids", "offsets", "structure_sha256", "graph_count", "atom_file", "line_file",
            "atom_sha256", "line_sha256", "provenance_sha256"):
        if metadata.get(key) != record.get(key):
            raise RuntimeError(f"structure-cache shard {shard} metadata field mismatch: {key}")
    result = {"shard": shard, "graph_count": record["graph_count"], "graph_files_loaded": False}
    if load_graphs:
        atom_graphs = dgl_module.load_graphs(str(atom_path))[0]
        line_graphs = dgl_module.load_graphs(str(line_path))[0]
        if len(atom_graphs) != record["graph_count"] or len(line_graphs) != record["graph_count"]:
            raise RuntimeError(f"structure-cache shard {shard} serialized graph count mismatch")
        result["graph_files_loaded"] = True
    return result


def verify_cache(cache_root: Path, expected: dict, *, dgl_module, load_graphs: bool = True) -> dict:
    cache_root = Path(cache_root)
    manifest_path = cache_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError("verified structure-cache PASS manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, expected)
    results = [verify_shard(cache_root, row, dgl_module=dgl_module, load_graphs=load_graphs)
        for row in manifest["shards"]]
    return {"status": "passed", "verified_shards": len(results),
        "verified_graphs": sum(row["graph_count"] for row in results), "manifest": manifest}


def quarantine_shard(cache_root: Path, quarantine_root: Path, shard: int, reason: str,
        *, timestamp: str | None = None) -> Path | None:
    cache_root, quarantine_root = Path(cache_root).resolve(), Path(quarantine_root).resolve()
    if cache_root == quarantine_root:
        raise RuntimeError("cache and quarantine roots must differ")
    timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = quarantine_root / f"shard_{shard:05d}_{timestamp}_{uuid.uuid4().hex[:8]}"
    candidates = []
    for pattern in (f"atom_{shard:05d}.bin*", f"line_{shard:05d}.bin*", f"shard_{shard:05d}.json*"):
        candidates.extend(path for path in cache_root.glob(pattern) if path.is_file())
    if not candidates:
        return None
    destination.mkdir(parents=True, exist_ok=False)
    for path in sorted(set(candidates)):
        os.replace(path, destination / path.name)
    (destination / "QUARANTINE_REASON.json").write_text(json.dumps({"reason": reason, "shard": shard,
        "quarantined_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def atomic_write_shard(cache_root: Path, shard: int, atom_graphs: list, line_graphs: list,
        ids: list[str], structure_hashes: list[str], *, provenance_sha256: str, dgl_module) -> dict:
    if not ids or len(atom_graphs) != len(ids) or len(line_graphs) != len(ids) or len(structure_hashes) != len(ids):
        raise RuntimeError("cannot write a malformed shard")
    cache_root = Path(cache_root)
    token = uuid.uuid4().hex
    atom_name, line_name, meta_name = f"atom_{shard:05d}.bin", f"line_{shard:05d}.bin", f"shard_{shard:05d}.json"
    atom_tmp = cache_root / f"{atom_name}.tmp.{token}"
    line_tmp = cache_root / f"{line_name}.tmp.{token}"
    meta_tmp = cache_root / f"{meta_name}.tmp.{token}"
    dgl_module.save_graphs(str(atom_tmp), atom_graphs)
    dgl_module.save_graphs(str(line_tmp), line_graphs)
    metadata = {"schema_version": SCHEMA_VERSION, "shard": shard, "ids": list(ids),
        "offsets": list(range(len(ids))), "structure_sha256": list(structure_hashes),
        "graph_count": len(ids), "atom_file": atom_name, "line_file": line_name,
        "provenance_sha256": provenance_sha256, "atom_sha256": sha256_file(atom_tmp),
        "line_sha256": sha256_file(line_tmp)}
    meta_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record = dict(metadata)
    record.update({"metadata_file": meta_name, "metadata_sha256": sha256_file(meta_tmp)})
    # Metadata is promoted last, so an interruption can never make a partial
    # graph pair appear complete. The next resume quarantines every remnant.
    os.replace(atom_tmp, cache_root / atom_name)
    os.replace(line_tmp, cache_root / line_name)
    os.replace(meta_tmp, cache_root / meta_name)
    return record


def record_from_metadata(cache_root: Path, shard: int) -> dict:
    meta_path = Path(cache_root) / f"shard_{shard:05d}.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    record = dict(metadata)
    record.update({"metadata_file": meta_path.name, "metadata_sha256": sha256_file(meta_path)})
    return record


def rebuild_invalid_shard(cache_root: Path, quarantine_root: Path, shard: int, *, atom_graphs: list,
        line_graphs: list, ids: list[str], structure_hashes: list[str], provenance_sha256: str,
        dgl_module) -> tuple[dict, Path | None]:
    """Quarantine any invalid/partial evidence, then atomically rebuild it."""
    cache_root = Path(cache_root)
    quarantine_destination = None
    try:
        current = record_from_metadata(cache_root, shard)
        if current.get("provenance_sha256") != provenance_sha256:
            raise RuntimeError("shard provenance mismatch")
        verify_shard(cache_root, current, dgl_module=dgl_module, load_graphs=True)
        return current, None
    except Exception as error:
        quarantine_destination = quarantine_shard(cache_root, quarantine_root, shard,
            f"rebuild requested after invalid/incomplete shard: {error}")
    record = atomic_write_shard(cache_root, shard, atom_graphs, line_graphs, ids, structure_hashes,
        provenance_sha256=provenance_sha256, dgl_module=dgl_module)
    verify_shard(cache_root, record, dgl_module=dgl_module, load_graphs=True)
    return record, quarantine_destination


def load_selected(cache_root: Path, structure_ids: list[str], expected: dict | None = None,
        *, return_stats: bool = False):
    import dgl
    cache_root = Path(cache_root)
    started = time.perf_counter()
    manifest = json.loads((cache_root / MANIFEST_NAME).read_text(encoding="utf-8"))
    validate_manifest(manifest, expected)
    lookup_finished = time.perf_counter()
    missing = [item for item in structure_ids if item not in manifest["entries"]]
    if missing:
        raise RuntimeError(f"structure cache lacks {len(missing)} requested IDs; first={missing[0]}")
    grouped = defaultdict(list)
    for output_index, structure_id in enumerate(structure_ids):
        row = manifest["entries"][structure_id]
        grouped[int(row["shard"])].append((output_index, int(row["offset"])))
    atoms, lines = [None] * len(structure_ids), [None] * len(structure_ids)
    shard_records = {int(row["shard"]): row for row in manifest["shards"]}
    hash_seconds = load_seconds = 0.0
    hits = misses = 0
    for shard, requests in grouped.items():
        record = shard_records[shard]
        hash_started = time.perf_counter()
        verify_shard(cache_root, record, dgl_module=dgl, load_graphs=False)
        hash_seconds += time.perf_counter() - hash_started
        memory_key = (str(cache_root.resolve()), manifest["provenance"]["provenance_sha256"], shard)
        if memory_key not in _SHARD_MEMORY_CACHE:
            misses += 1
            load_started = time.perf_counter()
            atom_path, line_path = cache_root / record["atom_file"], cache_root / record["line_file"]
            atom_graphs, line_graphs = dgl.load_graphs(str(atom_path))[0], dgl.load_graphs(str(line_path))[0]
            load_seconds += time.perf_counter() - load_started
            if len(atom_graphs) != record["graph_count"] or len(line_graphs) != record["graph_count"]:
                raise RuntimeError(f"structure-cache shard {shard} serialized graph count mismatch")
            _SHARD_MEMORY_CACHE[memory_key] = atom_graphs, line_graphs
        else:
            hits += 1
        atom_graphs, line_graphs = _SHARD_MEMORY_CACHE[memory_key]
        for output_index, offset in requests:
            atoms[output_index], lines[output_index] = atom_graphs[offset], line_graphs[offset]
    stats = {"requested_structures": len(structure_ids), "requested_shards": len(grouped),
        "manifest_lookup_seconds": lookup_finished - started, "hash_verification_seconds": hash_seconds,
        "graph_loading_seconds": load_seconds, "total_verified_load_seconds": time.perf_counter() - started,
        "memory_cache_hits": hits, "memory_cache_misses": misses, "verified_shards": len(grouped)}
    return (atoms, lines, stats) if return_stats else (atoms, lines)
