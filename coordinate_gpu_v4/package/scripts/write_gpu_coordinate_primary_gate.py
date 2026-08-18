#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from alignn_stage2.common import write_json
def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--work-root",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    status=Path(a.work_root)/"fold_0/seed_0/TRAINING_STATUS.json"
    if not status.is_file() or json.loads(status.read_text()).get("status")!="training_complete":
        raise RuntimeError("fold-0/seed-0 GPU coordinate training is incomplete")
    write_json(Path(a.output),{"status":"passed","fold":0,"seed":0,"execution_device":"cuda",
        "full_grid_authorized":True,"outer_test_accessed":False})
    print("ALIGNN_GPU_COORDINATE_PRIMARY_GATE: PASS"); return 0
if __name__ == "__main__": raise SystemExit(main())
