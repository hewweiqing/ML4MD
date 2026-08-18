"""Fail-closed CUDA runtime contract for the prospective A100 coordinate workflow."""
from __future__ import annotations

import os
import platform

import torch

DEVICE = torch.device("cuda")


def configure_gpu_runtime() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU-coordinate scientific execution")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"exactly one visible CUDA device is required, found {torch.cuda.device_count()}")
    name = torch.cuda.get_device_name(0)
    if "A100" not in name.upper():
        raise RuntimeError(f"an NVIDIA A100 is required, found {name!r}")
    torch.cuda.set_device(0)
    probe = torch.ones((4, 4), device=DEVICE) @ torch.ones((4, 4), device=DEVICE)
    if not torch.isfinite(probe).all():
        raise RuntimeError("non-finite CUDA runtime probe")
    return {
        "status": "passed", "selected_device": "cuda", "gpu_model": name,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_is_available": True, "torch_cuda_runtime": torch.version.cuda,
        "representative_tensor_device": str(probe.device), "python": platform.python_version(),
        "torch": torch.__version__,
    }


def assert_model_gpu(model: torch.nn.Module) -> None:
    devices = {str(value.device) for value in [*model.parameters(), *model.buffers()]}
    if devices != {"cuda:0"}:
        raise RuntimeError(f"model is not entirely on cuda:0: {sorted(devices)}")
