from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

DATASET_SHA256 = "9a028ed5750a4c76ca36e9f3c8d48fe0bf3fb21b76ec2289e58ae7048d527919"
ALIGNN_COMMIT = "f2366daa3413d28a825b46e34d001b5549b05a40"
MATBENCH_COMMIT = "936176db18ca4cd7b38cbd957c017a5bac770c6b"
FOLDS = tuple(range(5))
SEEDS = tuple(range(5))
CONDITIONS = ("A_Control_raw", "B_Control_temperature_scaled", "C_Random2_Descriptor_raw", "D_Random2_Descriptor_temperature_scaled")
MANIFEST_NAME = "PACKAGE_MANIFEST.json"
EXCLUSION_RULES = (
    "exact path PACKAGE_MANIFEST.json (manifest is not self-hashed)",
    "directory component in: __pycache__, .pytest_cache, logs, preflight, runtime_outputs, outputs, results, checkpoints, graph_cache, environment_reports",
    "suffix in: .pyc, .pyo, .tmp, .part",
    "basename matches setup*.log",
    "suffix .json.gz (downloaded dataset/runtime data)",
)
GENERATED_DIR_NAMES = {"__pycache__", ".pytest_cache", "logs", "preflight", "runtime_outputs", "outputs", "results", "checkpoints", "graph_cache", "environment_reports"}
GENERATED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".part"}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_ids(values: Iterable[Any]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def exclusion_reason(relative_path: str) -> str | None:
    parts = tuple(Path(relative_path).parts)
    name = parts[-1] if parts else relative_path
    lower_name = name.lower()
    if relative_path == MANIFEST_NAME:
        return "self-referential manifest excluded"
    generated_component = next((part for part in parts if part in GENERATED_DIR_NAMES), None)
    if generated_component:
        return f"generated/runtime directory component: {generated_component}"
    if Path(name).suffix.lower() in GENERATED_SUFFIXES:
        return f"generated/temporary suffix: {Path(name).suffix.lower()}"
    if lower_name.startswith("setup") and lower_name.endswith(".log"):
        return "setup log"
    if lower_name.endswith(".json.gz"):
        return "downloaded dataset/runtime data"
    return None


def scan_package(root: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    root = Path(root)
    candidates = []
    excluded = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        reason = exclusion_reason(relative)
        if reason:
            excluded.append({"path": relative, "reason": reason})
        else:
            candidates.append((relative, path))
    files = [{"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for relative, path in sorted(candidates, key=lambda item: item[0])]
    return files, sorted(excluded, key=lambda item: item["path"])


def canonical_manifest_lines(files: Iterable[dict[str, Any]]) -> list[str]:
    return [f"{item['sha256']}  {item['path']}\n" for item in sorted(files, key=lambda item: item["path"])]


def aggregate_from_lines(lines: Iterable[str]) -> str:
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def package_manifest(root: str | Path) -> dict[str, Any]:
    files, excluded = scan_package(root)
    lines = canonical_manifest_lines(files)
    return {"schema_version": 2,
        "algorithm": "SHA-256 of exact UTF-8 concatenation of lexically sorted canonical POSIX lines: <sha256>  <relative-path>\\n",
        "aggregate_sha256": aggregate_from_lines(lines), "files": files,
        "canonical_inputs": [line.rstrip("\n") for line in lines],
        "exclusion_rules": list(EXCLUSION_RULES), "excluded_at_generation": excluded}


def verify_manifest(root: str | Path, manifest_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(root)
    manifest_path = Path(manifest_path) if manifest_path else root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_files, excluded = scan_package(root)
    expected_by_path = {item["path"]: item for item in manifest["files"]}
    actual_by_path = {item["path"]: item for item in actual_files}
    missing = sorted(set(expected_by_path) - set(actual_by_path))
    unexpected = sorted(set(actual_by_path) - set(expected_by_path))
    mismatches = []
    for path in sorted(set(expected_by_path) & set(actual_by_path)):
        expected, actual = expected_by_path[path], actual_by_path[path]
        if expected["sha256"] != actual["sha256"] or expected.get("size_bytes") != actual.get("size_bytes"):
            mismatches.append({"path": path, "expected_sha256": expected["sha256"],
                "actual_sha256": actual["sha256"], "expected_size_bytes": expected.get("size_bytes"),
                "actual_size_bytes": actual.get("size_bytes")})
    expected_lines = canonical_manifest_lines(manifest["files"])
    actual_lines = canonical_manifest_lines(actual_files)
    declared_aggregate = manifest["aggregate_sha256"]
    expected_aggregate = aggregate_from_lines(expected_lines)
    actual_aggregate = aggregate_from_lines(actual_lines)
    passed = not missing and not unexpected and not mismatches and declared_aggregate == expected_aggregate == actual_aggregate
    return {"status": "passed" if passed else "failed", "declared_aggregate": declared_aggregate,
        "expected_aggregate": expected_aggregate, "actual_aggregate": actual_aggregate,
        "missing_declared_files": missing, "unexpected_immutable_files": unexpected,
        "individual_file_mismatches": mismatches,
        "ordered_expected_aggregate_inputs": [line.rstrip("\n") for line in expected_lines],
        "ordered_actual_aggregate_inputs": [line.rstrip("\n") for line in actual_lines],
        "excluded_paths": excluded, "exclusion_rules": list(EXCLUSION_RULES)}
