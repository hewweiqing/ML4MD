#!/usr/bin/env python3
"""Create explicit mutable manual approval for a measured CPU profile."""
import argparse, json, os
from pathlib import Path
from alignn_stage2.common import sha256_file, write_json
def main():
    p=argparse.ArgumentParser(); p.add_argument("--profile",required=True); p.add_argument("--output",required=True); p.add_argument("--reviewer",required=True); p.add_argument("--approve",action="store_true"); a=p.parse_args()
    value=json.loads(Path(a.profile).read_text(encoding="utf-8"))
    if not a.approve or value.get("status")!="passed" or value.get("measured_optimizer_batches")!=100: raise RuntimeError("explicit approval of a passing 100-batch profile is required")
    write_json(Path(a.output),{"status":"approved","reviewer":a.reviewer,"profile_sha256":sha256_file(Path(a.profile)),
        "coordinate_authorization_sha256":value["coordinate_authorization_sha256"],"slurm_job_id":os.getenv("SLURM_JOB_ID"),
        "acknowledged_24_hour_resumable_cpu_execution":True})
    print("CPU_PROFILE_APPROVAL: WRITTEN")
if __name__ == "__main__": main()
