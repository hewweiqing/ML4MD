#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from alignn_stage2.common import sha256_file, write_json

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--profile",required=True); p.add_argument("--output",required=True)
    p.add_argument("--reviewer",required=True); p.add_argument("--decision",choices=("approve","reject"),required=True)
    p.add_argument("--reason",required=True); a=p.parse_args(); path=Path(a.profile); value=json.loads(path.read_text())
    checks={"status":value.get("status")=="passed", "device":value.get("execution_device")=="cuda",
        "batches":value.get("measured_optimizer_batches")==100, "finite":value.get("finite_losses") is True,
        "outer_test_unaccessed":value.get("outer_test_accessed") is False,
        "runtime_within_24h":float(value.get("projected_paired_40_epoch_hours",1e9))<24.0,
        "peak_cuda_below_35gb":int(value.get("peak_cuda_memory_reserved_bytes",1<<60))<35*1024**3}
    approved=a.decision=="approve" and all(checks.values())
    report={"status":"approved" if approved else "rejected", "reviewer":a.reviewer,
        "reason":a.reason, "profile_sha256":sha256_file(path), "checks":checks,
        "decision_used_scientific_outcomes":False}
    write_json(Path(a.output),report)
    if not approved: raise RuntimeError(f"GPU coordinate profile not approved: {checks}")
    print("GPU COORDINATE PROFILE REVIEW: APPROVED"); return 0
if __name__ == "__main__": raise SystemExit(main())
