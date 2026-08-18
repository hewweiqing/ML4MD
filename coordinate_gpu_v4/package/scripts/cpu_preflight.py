#!/usr/bin/env python3
"""CPU-only package/runtime preflight; intentionally independent of sigma and dataset."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from alignn.models.alignn import ALIGNN
from alignn_stage2.common import sha256_file, write_json
from alignn_stage2.cpu_coordinate_training import model_config
from alignn_stage2.cpu_runtime import assert_model_cpu, configure_cpu_runtime

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True); args = parser.parse_args()
    runtime = configure_cpu_runtime(); model = ALIGNN(model_config()).to(torch.device("cpu")); assert_model_cpu(model)
    package = Path(__file__).resolve().parents[1]
    manifest = json.loads((package / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    report = {"status": "passed", "non_scientific": True, "dataset_accessed": False,
        "sigma_required": False, "execution_device": "cpu", "runtime": runtime,
        "parameter_devices": sorted({str(value.device) for value in model.parameters()}),
        "package_aggregate_sha256": manifest["aggregate_sha256"],
        "cpu_full_config_sha256": sha256_file(package / "CPU_FULL_CONFIG.json")}
    write_json(Path(args.output), report); print("ALIGNN_CPU_PREFLIGHT: PASS"); return 0
if __name__ == "__main__": raise SystemExit(main())
