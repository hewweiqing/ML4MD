#!/usr/bin/env python3
import json
from pathlib import Path

from alignn_stage2.slurm_resources import PARTITION_LIMITS_MB_PER_CPU, audit_slurm_resources


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    records = audit_slurm_resources(root / "slurm")
    print(json.dumps({"status": "passed", "partition_limits_mb_per_cpu": PARTITION_LIMITS_MB_PER_CPU,
        "audited_job_count": len(records), "jobs": records}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
