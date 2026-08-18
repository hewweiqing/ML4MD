#!/usr/bin/env python3
"""Static release audit for every v29 CPU Slurm job and scientific entry point."""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REQUIRED={"#SBATCH --account=research-ME-mse","#SBATCH --partition=compute","#SBATCH --ntasks=1",
    "#SBATCH --cpus-per-task=8","#SBATCH --mem-per-cpu=3968M"}
PROHIBITED_SLURM=("--gres=gpu","--gpus","gpu-a100","device=\"cuda\"",".cuda()","autocast(","max_memory_allocated")
SCIENTIFIC=[ROOT/"alignn_stage2"/name for name in ("coordinate_cache.py","coordinate_noise.py","cpu_split.py",
    "cpu_runtime.py","cpu_prediction_schema.py","cpu_coordinate_training.py","cpu_coordinate_calibrate_export.py",
    "cpu_coordinate_oof.py","cpu_coordinate_final_audit.py","calibration_contract.py")]

def audit() -> dict:
    failures=[]; jobs=sorted((ROOT/"slurm").glob("*.sbatch"))
    if len(jobs)!=10: failures.append(f"expected 10 CPU sbatch files, found {len(jobs)}")
    for path in jobs:
        text=path.read_text(encoding="utf-8")
        for directive in REQUIRED:
            if directive not in text: failures.append(f"{path.name}: missing {directive}")
        for token in PROHIBITED_SLURM:
            if token.lower() in text.lower(): failures.append(f"{path.name}: prohibited {token}")
        if 'export CUDA_VISIBLE_DEVICES=""' not in text: failures.append(f"{path.name}: CPU visibility guard missing")
        if "#SBATCH --nodes" in text: failures.append(f"{path.name}: unnecessary --nodes")
        match=re.search(r"#SBATCH --array=(.+)",text)
        if match and not match.group(1).endswith("%5"): failures.append(f"{path.name}: array concurrency is not 5")
    for path in SCIENTIFIC:
        text=path.read_text(encoding="utf-8").lower()
        tokens=(".cuda(","device=\"cuda\"","device='cuda'","autocast(")
        if path.name != "cpu_runtime.py": tokens += ("torch.cuda.",)
        for token in tokens:
            if token in text: failures.append(f"{path.name}: prohibited scientific CPU token {token}")
    return {"status":"passed" if not failures else "failed","audited_sbatch_files":[p.name for p in jobs],
        "audited_scientific_paths":[p.name for p in SCIENTIFIC],"failures":failures}

if __name__=="__main__":
    result=audit(); print(json.dumps(result,indent=2,sort_keys=True)); raise SystemExit(0 if result["status"]=="passed" else 1)
