#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--evidence",required=True);a=p.parse_args();v=json.loads(Path(a.evidence).read_text())
 expected="runtime target mismatch: expected {'python': '3.10.20', 'pip': '25.3', 'platform_system': 'Linux', 'platform_machine': 'x86_64', 'glibc': '2.35'}, found {'python': '3.10.20', 'pip': '25.3', 'platform_system': 'Linux', 'platform_machine': 'x86_64', 'glibc': '2.28'}"
 ok=(v.get("status")=="failed" and v.get("errors")==[expected] and v.get("target",{}).get("glibc")=="2.28" and v.get("pip_check")=="No broken requirements found." and v.get("undeclared_distributions")==[] and v.get("absent_distributions")==[] and v.get("version_mismatches")=={} and all(v.get("imports",{}).get(n,{}).get("passed") for n in ("torch","dgl","alignn")))
 print(json.dumps({"status":"passed" if ok else "failed"},indent=2));return 0 if ok else 1
if __name__=="__main__": raise SystemExit(main())
