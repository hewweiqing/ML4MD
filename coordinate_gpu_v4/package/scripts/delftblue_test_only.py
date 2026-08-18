#!/usr/bin/env python3
"""Run every DelftBlue static submission simulation; never submit a job."""
import json
from pathlib import Path

from alignn_stage2.slurm_live_validation import run_test_only_validation


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    print("SIMULATION ONLY: sbatch --test-only returns simulated job IDs and submits no jobs.")
    records = run_test_only_validation(root / "slurm")
    print(json.dumps({"status": "passed", "jobs_submitted": 0,
        "test_only_job_ids_are_simulations": True, "simulations": records}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
