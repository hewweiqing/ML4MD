#!/usr/bin/env python3
"""A100 runtime certification using one label-free dataset structure."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path

from alignn_stage2.cuda_runtime import inspect_runtime


def run_command(command: list[str]) -> dict:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {"command": command, "returncode": completed.returncode,
        "stdout": completed.stdout, "stderr": completed.stderr}


def write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--certification-output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--writable-root", required=True)
    parser.add_argument("--login-certification", required=True)
    parser.add_argument("--package-manifest", required=True)
    args = parser.parse_args()
    started = time.time()
    checks, errors = {}, []
    login_certification_path = Path(args.login_certification).resolve()
    package_manifest_path = Path(args.package_manifest).resolve()
    login_certification = json.loads(login_certification_path.read_text(encoding="utf-8"))
    package_manifest = json.loads(package_manifest_path.read_text(encoding="utf-8"))
    login_sha256 = hashlib.sha256(login_certification_path.read_bytes()).hexdigest()
    checks["login_certification_v20"] = (login_certification.get("certification_workflow_version") == 20
        and login_certification.get("full_pytest_status") == "passed"
        and login_certification.get("package_aggregate_sha256") == package_manifest.get("aggregate_sha256"))
    if not checks["login_certification_v20"]:
        errors.append("hash-bound v20 login certification is not passed")
    environment_prefix = str(Path(sys.prefix).resolve())
    cuda_runtime = inspect_runtime(Path(sys.prefix))
    checks["cuda_runtime_all_required_libraries"] = cuda_runtime["passed"]
    errors.extend(cuda_runtime["errors"])
    nvidia_smi = run_command(["nvidia-smi"])
    nvidia_query = run_command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    checks["nvidia_smi_succeeded"] = nvidia_smi["returncode"] == 0
    if not checks["nvidia_smi_succeeded"]:
        errors.append("nvidia-smi failed")
    gpu_model = driver_version = None
    if nvidia_query["returncode"] == 0 and nvidia_query["stdout"].strip():
        fields = [value.strip() for value in nvidia_query["stdout"].splitlines()[0].split(",")]
        gpu_model = fields[0] if fields else None
        driver_version = fields[1] if len(fields) > 1 else None
    checks["gpu_is_a100"] = bool(gpu_model and "A100" in gpu_model.upper())
    if not checks["gpu_is_a100"]:
        errors.append(f"A100 required, found {gpu_model}")

    torch_version = torch_cuda_runtime = None
    cuda_available = False
    gpu_tensor_operation = None
    try:
        import torch
        torch_version = torch.__version__
        torch_cuda_runtime = torch.version.cuda
        cuda_available = bool(torch.cuda.is_available())
        checks["torch_version"] = torch_version == "2.0.1+cu118"
        checks["torch_cuda_runtime"] = torch_cuda_runtime == "11.8"
        checks["torch_cuda_available"] = cuda_available
        if not checks["torch_version"]:
            errors.append(f"torch expected 2.0.1+cu118, found {torch_version}")
        if not checks["torch_cuda_runtime"]:
            errors.append(f"torch CUDA runtime expected 11.8, found {torch_cuda_runtime}")
        if not cuda_available:
            errors.append("torch.cuda.is_available() is false")
        if cuda_available:
            tensor = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
            result = tensor @ tensor.T
            torch.cuda.synchronize()
            gpu_tensor_operation = {"device": str(result.device), "shape": list(result.shape),
                "sum": float(result.sum().cpu()), "finite": bool(torch.isfinite(result).all().cpu())}
            checks["gpu_tensor_operation"] = gpu_tensor_operation["finite"] and gpu_tensor_operation["device"].startswith("cuda")
    except Exception as error:
        checks["torch_import_and_gpu_operation"] = False
        errors.append(f"torch runtime failed: {type(error).__name__}: {error}")

    cusparse_record = next(row for row in cuda_runtime["libraries"]
        if row["requested_soname"] == "libcusparse.so.11")
    libcusparse_path = cusparse_record["resolved_path"]
    libcusparse_error = cusparse_record["loader_error"]
    checks["libcusparse_resolved"] = cusparse_record["loader_succeeded"]

    dgl_version = dgl_error = None
    try:
        import dgl
        dgl_version = dgl.__version__
        checks["dgl_import"] = True
        checks["dgl_version"] = dgl_version == "1.1.1+cu118"
        if not checks["dgl_version"]:
            errors.append(f"DGL expected 1.1.1+cu118, found {dgl_version}")
        if cuda_available:
            graph = dgl.graph(([0, 1], [1, 0])).to("cuda")
            checks["dgl_cuda_graph"] = str(graph.device).startswith("cuda")
    except Exception as error:
        dgl_error = f"{type(error).__name__}: {error}"
        checks["dgl_import"] = False
        errors.append(f"DGL import/runtime failed: {dgl_error}")

    alignn_version = alignn_error = None
    model_constructed = False
    try:
        from alignn.models.alignn import ALIGNN, ALIGNNConfig
        alignn_version = version("alignn")
        model = ALIGNN(ALIGNNConfig(name="alignn", alignn_layers=4, gcn_layers=4,
            atom_input_features=92, edge_input_features=80, triplet_input_features=40,
            embedding_features=64, hidden_features=256, classification=True,
            num_classes=2, link="identity"))
        model.to("cuda")
        model_constructed = True
        checks["alignn_version"] = alignn_version == "2025.4.1"
        checks["alignn_import_and_model_construction"] = True
        if not checks["alignn_version"]:
            errors.append(f"ALIGNN expected 2025.4.1, found {alignn_version}")
        import gzip
        import ijson
        import dgl
        from alignn.graphs import Graph
        from jarvis.core.atoms import pmg_to_atoms
        from matbench.bench import MatbenchBenchmark
        from pymatgen.core import Structure
        benchmark = MatbenchBenchmark(autoload=False, subset=["matbench_mp_is_metal"])
        task = next(iter(benchmark.tasks))
        checks["matbench_task_and_five_folds"] = len(task.folds_map) == 5
        with gzip.open(args.dataset, "rb") as stream:
            first = next(ijson.items(stream, "data.item.item", use_float=True))
        if not isinstance(first, dict):
            raise RuntimeError("first dataset item is not a structure dictionary")
        atom, line = Graph.atom_dgl_multigraph(atoms=pmg_to_atoms(Structure.from_dict(first)),
            neighbor_strategy="k-nearest", cutoff=8.0, max_neighbors=12, atom_features="cgcnn",
            compute_line_graph=True, use_canonize=True, id="environment-gate")
        batch = (dgl.batch([atom]).to("cuda"), dgl.batch([line]).to("cuda"), None)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        if output.ndim == 1 and output.shape[0] == 2:
            output = output.unsqueeze(0)
        if tuple(output.shape) != (1, 2):
            raise RuntimeError(f"singleton ALIGNN output must normalize to [1,2], found {tuple(output.shape)}")
        loss = torch.nn.functional.nll_loss(output, torch.tensor([0], device="cuda"))
        loss.backward(); optimizer.step(); torch.cuda.synchronize()
        checks["one_graph_forward_backward_optimizer_step"] = bool(torch.isfinite(loss).cpu())
        probe = Path(args.writable_root) / ".alignn_environment_write_probe"
        probe.parent.mkdir(parents=True, exist_ok=True); probe.write_text("ok\n", encoding="utf-8"); probe.unlink()
        checks["persistent_output_writable"] = True
    except Exception as error:
        alignn_error = f"{type(error).__name__}: {error}"
        checks["alignn_import_and_model_construction"] = False
        errors.append(f"ALIGNN import/model construction failed: {alignn_error}")

    required = ("login_certification_v20", "cuda_runtime_all_required_libraries", "nvidia_smi_succeeded", "gpu_is_a100", "torch_version", "torch_cuda_runtime",
        "torch_cuda_available", "gpu_tensor_operation", "libcusparse_resolved", "dgl_import",
        "dgl_version", "dgl_cuda_graph", "alignn_version", "alignn_import_and_model_construction",
        "matbench_task_and_five_folds", "one_graph_forward_backward_optimizer_step", "persistent_output_writable")
    passed = not errors and all(checks.get(name) is True for name in required)
    component_versions = {}
    for package_name in ("nvidia-cuda-runtime-cu11", "nvidia-cublas-cu11",
            "nvidia-cusparse-cu11", "nvidia-cusolver-cu11"):
        try:
            component_versions[package_name] = version(package_name)
        except Exception as error:
            component_versions[package_name] = f"unavailable: {type(error).__name__}: {error}"
            errors.append(f"runtime component {package_name} version unavailable: {error}")
    passed = passed and not errors
    report = {"schema_version": 5, "environment_revision": 9,
        "environment_prefix": environment_prefix,
        "package_aggregate_sha256": package_manifest.get("aggregate_sha256"),
        "login_certification_sha256": login_sha256,
        "status": "passed" if passed else "failed",
        "a100_runtime_certified": passed, "scientific_training_launched": False,
        "dataset_structure_accessed": True, "dataset_labels_accessed": False,
        "outer_test_outcomes_accessed": False, "temperature_scaler_required": False,
        "structure_access_policy": "OUTER_TEST_STRUCTURE_PREPROCESSING_POLICY.md",
        "hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version,
        "gpu_model": gpu_model, "nvidia_driver_version": driver_version,
        "nvidia_smi": nvidia_smi, "nvidia_smi_query": nvidia_query,
        "torch_version": torch_version, "torch_cuda_runtime": torch_cuda_runtime,
        "cuda_runtime_component_versions": component_versions,
        "cuda_runtime_library_report": cuda_runtime,
        "torch_cuda_is_available": cuda_available, "gpu_tensor_operation": gpu_tensor_operation,
        "libcusparse_path": libcusparse_path, "libcusparse_error": libcusparse_error,
        "dgl_version": dgl_version, "dgl_error": dgl_error,
        "alignn_version": alignn_version, "alignn_error": alignn_error,
        "alignn_model_constructed_on_cuda": model_constructed,
        "checks": checks, "errors": errors, "elapsed_seconds": time.time() - started}
    write_atomic(Path(args.output), report)
    if passed:
        write_atomic(Path(args.certification_output), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise
