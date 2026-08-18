#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from alignn_stage2.common import sha256_file

def load(path):
    p=Path(path)
    if not p.is_file(): raise RuntimeError(f"required gate missing: {p}")
    return p,json.loads(p.read_text())
def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--preflight",required=True); p.add_argument("--profile")
    p.add_argument("--approval"); p.add_argument("--primary-gate"); a=p.parse_args()
    _,pre=load(a.preflight)
    if pre.get("status")!="passed" or pre.get("outer_test_accessed") is not False: raise RuntimeError("A100 preflight invalid")
    if a.profile or a.approval:
        if not (a.profile and a.approval): raise RuntimeError("profile and approval must be supplied together")
        pp,profile=load(a.profile); _,approval=load(a.approval)
        if profile.get("status")!="passed" or profile.get("execution_device")!="cuda": raise RuntimeError("GPU profile invalid")
        if approval.get("status")!="approved" or approval.get("profile_sha256")!=sha256_file(pp): raise RuntimeError("GPU profile approval invalid")
    if a.primary_gate:
        _,gate=load(a.primary_gate)
        if gate.get("status")!="passed" or gate.get("full_grid_authorized") is not True: raise RuntimeError("primary GPU coordinate gate invalid")
    print("REQUIRED_GPU_COORDINATE_GATES: PASS"); return 0
if __name__ == "__main__": raise SystemExit(main())
