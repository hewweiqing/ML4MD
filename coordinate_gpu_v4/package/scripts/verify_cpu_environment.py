#!/usr/bin/env python3
"""Verify the reused certified Python environment without installing or selecting an accelerator."""
import json, os, platform, subprocess, sys
from importlib.metadata import version
from pathlib import Path
def main():
    expected={"python":"3.10","alignn":"2025.4.1","dgl":"1.1.1+cu118","torch":"2.0.1+cu118","numpy":"1.26.4",
        "matbench":"0.6","matminer":"0.9.3","monty":"2025.3.3","pyparsing":"2.4.7"}
    actual={"python":".".join(map(str,sys.version_info[:2]))}
    for name in expected:
        if name!="python": actual[name]=version(name)
    errors={key:{"expected":value,"actual":actual.get(key)} for key,value in expected.items() if actual.get(key)!=value}
    pip=subprocess.run([sys.executable,"-m","pip","check"],text=True,capture_output=True,check=False)
    if pip.returncode: errors["pip_check"]={"stdout":pip.stdout,"stderr":pip.stderr}
    report={"status":"passed" if not errors else "failed","execution_policy":"cpu_only","python":sys.executable,
        "platform":platform.platform(),"CUDA_VISIBLE_DEVICES":os.getenv("CUDA_VISIBLE_DEVICES"),"versions":actual,
        "pip_check":pip.stdout.strip(),"errors":errors}
    print(json.dumps(report,indent=2,sort_keys=True)); raise SystemExit(1 if errors else 0)
if __name__=="__main__": main()
