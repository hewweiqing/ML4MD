#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import torch
from alignn.models.alignn import ALIGNN
from alignn_stage2.common import write_json
from alignn_stage2.gpu_coordinate_training import forward_raw, model_config
from alignn_stage2.gpu_runtime import DEVICE, assert_model_gpu, configure_gpu_runtime

def main() -> int:
    runtime = configure_gpu_runtime()
    model = ALIGNN(model_config()).to(DEVICE); assert_model_gpu(model)
    descriptors = torch.randn(8, model.fc.in_features, device=DEVICE)
    labels = torch.arange(8, device=DEVICE) % 2
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    logits = model.fc(descriptors)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward(); optimizer.step(); torch.cuda.synchronize()
    report = {"status": "passed", "non_scientific_resource_only": True,
        "outer_test_accessed": False, "runtime": runtime,
        "model_constructed_on_cuda": True, "classifier_optimizer_step": True,
        "finite_loss": bool(torch.isfinite(loss)),
        "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved())}
    write_json(Path("preflight/GPU_COORDINATE_A100_PREFLIGHT.json"), report)
    print("ALIGNN_GPU_COORDINATE_A100_PREFLIGHT: PASS")
    return 0
if __name__ == "__main__": raise SystemExit(main())
