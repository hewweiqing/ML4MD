#!/usr/bin/env python3
"""Login-node certification after the shared CUDA runtime bootstrap is active."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib import import_module
from importlib.metadata import distribution, distributions, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alignn_stage2.cuda_runtime import inspect_runtime
from alignn_stage2.dependency_resolution import normalize

EXPECTED = {
    "alignn": "2025.4.1", "dgl": "1.1.1+cu118", "flake8": "7.3.0",
    "jarvis-tools": "2026.6.12", "matbench": "0.6", "matminer": "0.9.3",
    "monty": "2025.3.3", "numpy": "1.26.4", "pycodestyle": "2.14.0",
    "pydocstyle": "6.3.0", "pymatgen": "2025.10.7", "pyparsing": "2.4.7",
    "torch": "2.0.1+cu118", "exceptiongroup": "1.3.1", "tomli": "2.4.1",
    "pip": "25.3", "setuptools": "80.9.0", "wheel": "0.45.1",
    "nvidia-cuda-runtime-cu11": "11.8.89", "nvidia-cublas-cu11": "11.11.3.6",
    "nvidia-cusolver-cu11": "11.4.1.48", "nvidia-cusparse-cu11": "11.7.5.86",
}


def inventory(name: str) -> dict:
    dist = distribution(name)
    files = list(dist.files or [])
    missing = [str(item) for item in files if not Path(dist.locate_file(item)).exists()]
    return {"declared_file_count": len(files), "missing_declared_files": missing}


def write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--constraints-sha256")
    parser.add_argument("--runtime-requirements-sha256", required=True)
    parser.add_argument("--runtime-lock-sha256", required=True)
    parser.add_argument("--python-lock-sha256", required=True)
    parser.add_argument("--bootstrap-requirements-sha256", required=True)
    parser.add_argument("--local-distributions-sha256", required=True)
    parser.add_argument("--resolution-evidence-sha256", required=True)
    parser.add_argument("--torch-requirement-sha256", required=True)
    parser.add_argument("--dgl-requirement-sha256", required=True)
    parser.add_argument("--expected-prefix", required=True)
    parser.add_argument("--allow-staging", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    errors, actual, inventories, imports = [], {}, {}, {}
    actual_prefix = str(Path(sys.prefix).resolve())
    expected_prefix = str(Path(args.expected_prefix).resolve())
    if actual_prefix != expected_prefix:
        errors.append(f"interpreter prefix mismatch: expected {expected_prefix}, found {actual_prefix}")
    expected_suffix = "/alignn_matbench_is_metal_cu118_v9.staging" if args.allow_staging else "/alignn_matbench_is_metal_cu118_v9"
    if not expected_prefix.endswith(expected_suffix):
        errors.append(f"revision-9 prefix suffix {expected_suffix} required, found {expected_prefix}")
    if platform.python_version() != "3.10.20":
        errors.append(f"Python must be exactly 3.10.20, found {platform.python_version()}")
    for name, expected in EXPECTED.items():
        try:
            actual[name] = version(name)
            if actual[name] != expected:
                errors.append(f"{name}: expected {expected}, found {actual[name]}")
        except Exception as error:
            actual[name] = None
            errors.append(f"{name} missing: {error}")
    for name in ("alignn", "dgl", "matbench", "nvidia-cusparse-cu11"):
        try:
            inventories[name] = inventory(name)
            if inventories[name]["missing_declared_files"]:
                errors.append(f"{name} distribution has missing installed files")
        except Exception as error:
            errors.append(f"{name} distribution inventory failed: {type(error).__name__}: {error}")
    closure = json.loads((ROOT / "DEPENDENCY_RESOLUTION_EVIDENCE.json").read_text(encoding="utf-8"))
    declared = {normalize(row["normalized_name"]): row["version"]
                for row in closure["wheels"] if row.get("selected_for_lock")}
    local = json.loads((ROOT / "DECLARED_LOCAL_DISTRIBUTIONS.json").read_text(encoding="utf-8"))
    declared.update({normalize(name): item["version"] for name, item in local["distributions"].items()})
    installed = {normalize(item.metadata["Name"]): item.version for item in distributions()
                 if item.metadata.get("Name")}
    undeclared = sorted(set(installed) - set(declared))
    absent = sorted(set(declared) - set(installed))
    wrong = {name: {"expected": declared[name], "actual": installed[name]}
             for name in sorted(set(declared) & set(installed)) if declared[name] != installed[name]}
    if undeclared:
        errors.append(f"installed distributions outside declared closure: {undeclared}")
    if absent:
        errors.append(f"declared distributions absent from installation: {absent}")
    if wrong:
        errors.append(f"installed distribution versions differ from closure: {wrong}")
    runtime = inspect_runtime(Path(expected_prefix))
    errors.extend(runtime["errors"])
    for module_name in ("torch", "dgl", "alignn"):
        try:
            module = import_module(module_name)
            imports[module_name] = {"passed": True, "version": getattr(module, "__version__", None)}
        except Exception as error:
            imports[module_name] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
            errors.append(f"import {module_name} failed: {imports[module_name]['error']}")
    torch_cuda = None
    if imports.get("torch", {}).get("passed"):
        import torch
        torch_cuda = torch.version.cuda
        if torch_cuda != "11.8":
            errors.append(f"torch CUDA metadata: expected 11.8, found {torch_cuda}")
    report = {
        "schema_version": 5, "environment_revision": 9,
        "status": "login_node_installation_certified" if not errors else "failed",
        "a100_runtime_certified": False, "gpu_required_for_this_check": False,
        "gpu_runtime_status": "loader_and_imports_certified_A100_execution_pending",
        "environment_prefix": actual_prefix, "expected_prefix": expected_prefix,
        "python": platform.python_version(), "versions": actual,
        "torch_cuda_runtime_metadata": torch_cuda, "imports": imports,
        "cuda_runtime": runtime, "distribution_file_inventories": inventories,
        "declared_distribution_count": len(declared),
        "installed_distribution_count": len(installed),
        "undeclared_distributions": undeclared,
        "absent_distributions": absent,
        "version_mismatches": wrong,
        "constraints_sha256": args.constraints_sha256,
        "runtime_requirements_sha256": args.runtime_requirements_sha256,
        "runtime_lock_sha256": args.runtime_lock_sha256, "errors": errors,
        "python_lock_sha256": args.python_lock_sha256,
        "bootstrap_requirements_sha256": args.bootstrap_requirements_sha256,
        "local_distributions_sha256": args.local_distributions_sha256,
        "resolution_evidence_sha256": args.resolution_evidence_sha256,
        "torch_requirement_sha256": args.torch_requirement_sha256,
        "dgl_requirement_sha256": args.dgl_requirement_sha256,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        write_atomic(Path(args.output), report)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
