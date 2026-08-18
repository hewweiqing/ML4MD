#!/usr/bin/env python3
import argparse
from pathlib import Path
from alignn_stage2.production import verify_complete

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--work-root",required=True);args=parser.parse_args();root=Path(args.work_root)
    failed=[f"fold_{f}/seed_{s}" for f in range(5) for s in range(5) if not verify_complete(root/f"fold_{f}"/f"seed_{s}",f,s)]
    if failed: raise RuntimeError(f"grid incomplete or hash-invalid: {failed}")
    print("ALIGNN_25_CELL_GRID: PASS");return 0
if __name__=="__main__":raise SystemExit(main())
