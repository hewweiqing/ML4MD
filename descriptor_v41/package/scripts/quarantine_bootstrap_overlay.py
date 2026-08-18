#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil, sys
from importlib.metadata import distributions
from pathlib import Path
EXPECTED={"pip":{"25.3","26.2.1"},"setuptools":{"80.9.0","84.0.0"},"wheel":{"0.45.1","0.47.0"}}
def main():
 p=argparse.ArgumentParser();p.add_argument("--prefix",required=True);p.add_argument("--quarantine",required=True);a=p.parse_args();prefix=Path(a.prefix).resolve();q=Path(a.quarantine).resolve()
 if Path(sys.prefix).resolve()!=prefix: raise SystemExit("target interpreter required")
 site=prefix/"lib/python3.10/site-packages";found={k:set() for k in EXPECTED}
 for d in distributions(path=[str(site)]):
  n=(d.metadata.get("Name") or "").lower()
  if n in found: found[n].add(d.version)
 if found!=EXPECTED: raise SystemExit(f"overlay differs: {found}")
 if q.exists(): raise SystemExit(f"quarantine exists: {q}")
 q.mkdir(parents=True); paths=[site/n for n in ("pip","setuptools","wheel","pkg_resources","_distutils_hack","distutils-precedence.pth")]
 for pat in ("pip-*.dist-info","setuptools-*.dist-info","setuptools-*.egg-info","wheel-*.dist-info"): paths.extend(site.glob(pat))
 paths.extend(prefix/"bin"/n for n in ("pip","pip3","pip3.10","wheel"));moved=[]
 for s in sorted(set(paths)):
  if not s.exists(): continue
  if prefix not in s.resolve().parents: raise SystemExit(f"outside prefix: {s}")
  d=q/("bin" if s.parent.name=="bin" else "site-packages")/s.name;d.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(s),str(d));moved.append(str(s))
 (q/"QUARANTINE_MANIFEST.json").write_text(json.dumps({"versions":{k:sorted(v) for k,v in found.items()},"moved":moved},indent=2)+"\n")
 print(json.dumps({"status":"passed","moved_count":len(moved),"quarantine":str(q)},indent=2));return 0
if __name__=="__main__": raise SystemExit(main())
