#!/usr/bin/env python3
"""Verify an external prospective sigma authority and copy it to mutable runtime evidence."""
from __future__ import annotations
import argparse
import os
import shutil
from pathlib import Path
from alignn_stage2.sigma_authorization import require_authorized_sigma

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); authority = require_authorized_sigma(authorization_path=args.input)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp"); shutil.copyfile(args.input, temporary); os.replace(temporary, output)
    confirmed = require_authorized_sigma(authorization_path=output)
    if confirmed.authorization_sha256 != authority.authorization_sha256: raise RuntimeError("authorization changed during import")
    print(f"COORDINATE_SIGMA_AUTHORIZATION: PASS {output}"); return 0
if __name__ == "__main__": raise SystemExit(main())
