#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from alignn_stage2.common import sha256_file
from alignn_stage2.sigma_authorization import require_authorized_sigma
def main():
    p=argparse.ArgumentParser(); p.add_argument("--profile",required=True); p.add_argument("--approval",required=True); a=p.parse_args()
    authority=require_authorized_sigma(); profile=Path(a.profile); approval=json.loads(Path(a.approval).read_text(encoding="utf-8"))
    if approval.get("status")!="approved" or approval.get("profile_sha256")!=sha256_file(profile) or approval.get("coordinate_authorization_sha256")!=authority.authorization_sha256: raise RuntimeError("current CPU profile is not approved")
    print("REQUIRED_CPU_PROFILE_APPROVAL: PASS")
if __name__ == "__main__": main()
