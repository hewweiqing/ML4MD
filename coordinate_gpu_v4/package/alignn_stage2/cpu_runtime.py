"""CPU-only DelftBlue runtime contract."""
from __future__ import annotations

import json
import os
import platform
import resource
from pathlib import Path

import torch


def configure_cpu_runtime(*, fail_if_cuda_visible: bool = True) -> dict:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if fail_if_cuda_visible and visible not in ("", "-1"):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty or -1 for CPU scientific execution")
    if fail_if_cuda_visible and torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() unexpectedly true in a CPU scientific job")
    allocated = max(1, int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    intra = min(allocated, 8)
    interop = min(2, max(1, intra // 4))
    torch.set_num_threads(intra)
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        if torch.get_num_interop_threads() != interop:
            raise
    probe = torch.ones((4, 4), device=torch.device("cpu")) @ torch.ones((4, 4), device=torch.device("cpu"))
    return {"status": "passed", "selected_device": "cpu", "CUDA_VISIBLE_DEVICES": visible,
        "torch_cuda_is_available": bool(torch.cuda.is_available()), "representative_tensor_device": str(probe.device),
        "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
        "slurm_cpus_per_task": os.getenv("SLURM_CPUS_PER_TASK"), "cpu_model": platform.processor(),
        "host_ram_bytes": _host_ram(), "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "python": platform.python_version(), "torch": torch.__version__, "dgl_backend": os.getenv("DGLBACKEND", "pytorch")}


def _host_ram() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES"); size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * size)
    except (AttributeError, ValueError, OSError):
        return None


def assert_model_cpu(model: torch.nn.Module) -> None:
    devices = {str(value.device) for value in [*model.parameters(), *model.buffers()]}
    if devices != {"cpu"}:
        raise RuntimeError(f"model is not entirely on CPU: {sorted(devices)}")


def write_runtime_evidence(path: str | Path, evidence: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)

