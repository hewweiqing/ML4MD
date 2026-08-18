#!/usr/bin/env python3
import json
from pathlib import Path

from alignn_stage2.common import verify_manifest


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report = verify_manifest(root)
    if report["status"] != "passed":
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit("package integrity failure; complete diagnostics printed above")
    print(json.dumps({"status": "passed", "expected_aggregate": report["expected_aggregate"],
        "actual_aggregate": report["actual_aggregate"],
        "individual_file_mismatches": report["individual_file_mismatches"],
        "missing_declared_files": report["missing_declared_files"],
        "unexpected_immutable_files": report["unexpected_immutable_files"],
        "excluded_path_count": len(report["excluded_paths"])}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
