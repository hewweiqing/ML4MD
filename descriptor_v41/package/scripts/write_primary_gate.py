#!/usr/bin/env python3
import argparse
from pathlib import Path
from alignn_stage2.common import write_json
from alignn_stage2.production import verify_complete


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--final-root", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    if not verify_complete(Path(args.final_root) / "fold_0" / "seed_0", 0, 0):
        raise RuntimeError("fold-0 seed-0 COMPLETE.json is not valid")
    write_json(args.output, {"status": "passed", "fold": 0, "seed": 0,
        "marker": "ALIGNN_FOLD0_SEED0_PRIMARY: PASS", "full_grid_authorized": True})
    return 0

if __name__ == "__main__": raise SystemExit(main())
