#!/usr/bin/env python3
"""Run DelftBlue sbatch --test-only simulations for every CPU job; submits nothing."""
import json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    print("SIMULATION ONLY: sbatch --test-only returns simulated IDs and submits no jobs.")
    rows=[]; failures=[]
    for path in sorted((ROOT/"slurm").glob("*.sbatch")):
        done=subprocess.run(["sbatch","--test-only",str(path)],cwd=ROOT,text=True,capture_output=True,check=False)
        combined=(done.stdout+"\n"+done.stderr).lower()
        bad=done.returncode!=0 or any(x in combined for x in ("error:","allocation failure","per-node basis","non-exclusive"))
        rows.append({"file":path.name,"returncode":done.returncode,"stdout":done.stdout.strip(),"stderr":done.stderr.strip(),"simulation_only":True})
        if bad: failures.append(path.name)
    result={"status":"failed" if failures else "passed","jobs_submitted":0,"test_only_job_ids_are_simulations":True,"failures":failures,"simulations":rows}
    print(json.dumps(result,indent=2)); raise SystemExit(1 if failures else 0)
if __name__=="__main__": main()
