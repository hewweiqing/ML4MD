#!/usr/bin/env python3
import argparse, json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--path", required=True); args = parser.parse_args()
    path = Path(args.path)
    if not path.is_file(): raise RuntimeError("primary fold-0 seed-0 gate marker is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "passed" or not value.get("full_grid_authorized"):
        raise RuntimeError("primary fold-0 seed-0 gate did not authorize the grid")
    return 0
if __name__ == "__main__": raise SystemExit(main())
