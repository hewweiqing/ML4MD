#!/usr/bin/env python3
"""Record exact resolved wheel metadata and emit the hash-locked Python dependency set."""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import platform
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from importlib.metadata import version

SEPARATE_INSTALLS = {
    "torch", "dgl", "nvidia-cuda-runtime-cu11", "nvidia-cublas-cu11",
    "nvidia-cusparse-cu11", "nvidia-cusolver-cu11",
}
BOOTSTRAP = {"pip", "setuptools", "wheel"}
LOCK_EXCLUDED = SEPARATE_INSTALLS | BOOTSTRAP


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_record(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = email.parser.BytesParser().parsebytes(archive.read(metadata_name))
    return {"name": metadata["Name"], "normalized_name": normalized(metadata["Name"]),
            "version": metadata["Version"], "filename": path.name, "size_bytes": path.stat().st_size,
            "sha256": digest(path), "requires_python": metadata.get("Requires-Python"),
            "requires_dist": metadata.get_all("Requires-Dist", [])}


def preference(record: dict) -> tuple:
    filename = record["filename"].lower()
    return ("manylinux2014_x86_64" in filename, "manylinux1_x86_64" not in filename, filename)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--python-lock-output", required=True)
    args = parser.parse_args()
    if (platform.python_version() != "3.10.20" or platform.system() != "Linux"
            or platform.machine() != "x86_64" or version("pip") != "25.3"):
        raise SystemExit("resolution must run under CPython 3.10.20 / Linux x86_64 / pip 25.3")
    records = [wheel_record(path) for path in sorted(Path(args.wheel_dir).glob("*.whl"))]
    if not records:
        raise SystemExit("no wheel files found")
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["normalized_name"]].append(record)
    if "nvidia-nvjitlink-cu11" in groups:
        raise SystemExit("invalid CUDA-12-style nvidia-nvjitlink-cu11 artifact present")
    selected = {}
    for name, candidates in groups.items():
        versions = {item["version"] for item in candidates}
        if len(versions) != 1:
            raise SystemExit(f"multiple versions resolved for {name}: {sorted(versions)}")
        selected[name] = max(candidates, key=preference)
    for record in records:
        record["selected_for_lock"] = selected[record["normalized_name"]]["filename"] == record["filename"]
    required = SEPARATE_INSTALLS | BOOTSTRAP | {"triton", "cmake", "lit", "filelock",
        "typing-extensions", "sympy", "networkx", "jinja2", "markupsafe", "requests",
        "exceptiongroup", "tomli"}
    missing = sorted(required - set(selected))
    if missing:
        raise SystemExit(f"required resolved distributions missing: {missing}")
    lock_lines = ["# Generated from DEPENDENCY_RESOLUTION_EVIDENCE.json; Linux x86_64 / CPython 3.10."]
    for name in sorted(set(selected) - LOCK_EXCLUDED):
        record = selected[name]
        lock_lines.append(f"{record['name']}=={record['version']} --hash=sha256:{record['sha256']}")
    Path(args.python_lock_output).write_text("\n".join(lock_lines) + "\n", encoding="utf-8")
    output = {"schema_version": 2, "status": "passed_static_resolution", "target": {
        "implementation": "cp", "python_version": "3.10", "python_full_version": "3.10.20",
        "pip_version": "25.3", "platform_system": "Linux", "platform_machine": "x86_64", "abi": "cp310",
        "platforms": ["linux_x86_64", "manylinux1_x86_64", "manylinux2014_x86_64",
            "manylinux_2_17_x86_64", "manylinux_2_28_x86_64"]},
        "indexes": ["https://download.pytorch.org/whl/cu118", "https://pypi.org/simple"],
        "dgl_url": "https://data.dgl.ai/wheels/cu118/dgl-1.1.1%2Bcu118-cp310-cp310-manylinux1_x86_64.whl",
        "resolved_distribution_count": len(selected), "downloaded_wheel_count": len(records),
        "invalid_nvjitlink_requirement_present": False, "wheels": records}
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "resolved": len(selected), "downloaded": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
