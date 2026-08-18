#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from alignn_stage2.dependency_resolution import validate_resolution

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    evidence = json.loads((ROOT / "DEPENDENCY_RESOLUTION_EVIDENCE.json").read_text(encoding="utf-8"))
    locks = [(ROOT / name).read_text(encoding="utf-8") for name in (
        "BOOTSTRAP_REQUIREMENTS.txt", "PYTHON_DEPENDENCY_LOCK.txt", "TORCH_CU118_REQUIREMENT.txt",
        "DGL_CU118_REQUIREMENT.txt", "CUDA11_RUNTIME_REQUIREMENTS.txt")]
    result = validate_resolution(evidence, locks)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
