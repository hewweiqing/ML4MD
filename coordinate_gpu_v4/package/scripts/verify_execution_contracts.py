#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from alignn_stage2.execution_contracts import run_execution_contracts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    print(json.dumps(run_execution_contracts(args.device), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
