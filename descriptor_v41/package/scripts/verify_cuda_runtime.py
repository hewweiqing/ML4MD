#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignn_stage2.cuda_runtime import inspect_runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--library-only", action="store_true")
    args = parser.parse_args()
    report = inspect_runtime(Path(args.prefix))
    if not args.library_only and report["passed"]:
        imports = {}
        for name in ("torch", "dgl", "alignn"):
            try:
                module = __import__(name)
                imports[name] = {"passed": True, "version": getattr(module, "__version__", None)}
            except Exception as error:
                imports[name] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
                report["errors"].append(f"import {name} failed: {imports[name]['error']}")
        report["imports"] = imports
        report["passed"] = not report["errors"]
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
