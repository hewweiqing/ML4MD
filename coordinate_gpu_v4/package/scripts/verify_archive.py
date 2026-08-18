#!/usr/bin/env python3
"""Read-only archive safety, duplication and normalized-permission verifier."""
import argparse, hashlib, json, tarfile
from pathlib import PurePosixPath
def main():
    p=argparse.ArgumentParser(); p.add_argument("archive"); p.add_argument("--expected-sha256"); a=p.parse_args()
    digest=hashlib.sha256(open(a.archive,"rb").read()).hexdigest(); failures=[]
    if a.expected_sha256 and digest!=a.expected_sha256: failures.append("archive SHA-256 mismatch")
    with tarfile.open(a.archive,"r:gz") as handle: members=handle.getmembers()
    names=[m.name for m in members]
    if len(names)!=len(set(names)): failures.append("duplicate archive members")
    for member in members:
        path=PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or "\\" in member.name: failures.append(f"unsafe path: {member.name}")
        executable=member.isdir() or member.name.endswith((".sh",".sbatch")) or ("/scripts/" in member.name and member.name.endswith(".py"))
        expected=0o755 if executable else 0o644
        if member.mode!=expected: failures.append(f"permission mismatch: {member.name}")
    print(json.dumps({"status":"failed" if failures else "passed","archive_sha256":digest,"member_count":len(members),"failures":failures},indent=2))
    raise SystemExit(1 if failures else 0)
if __name__=="__main__": main()
