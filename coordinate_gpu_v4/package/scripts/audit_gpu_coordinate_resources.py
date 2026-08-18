#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from alignn_stage2.slurm_resources import audit_slurm_resources

if __name__ == "__main__":
    rows = audit_slurm_resources(Path(__file__).resolve().parents[1] / "slurm")
    print(json.dumps({"status": "passed", "audited_jobs": len(rows), "jobs": rows}, indent=2))
