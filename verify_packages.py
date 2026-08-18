"""Verify the immutable package manifests from the repo root."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PACKAGES = (
    ROOT / "descriptor_v41" / "package",
    ROOT / "coordinate_gpu_v4" / "package",
    ROOT / "stage3_calibration" / "package",
)


def main() -> int:
    for package in PACKAGES:
        environment = os.environ.copy()
        prior = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(package) if not prior else str(package) + os.pathsep + prior
        )
        print(f"Verifying {package.relative_to(ROOT)}", flush=True)
        subprocess.run(
            [sys.executable, "scripts/verify_package.py"],
            cwd=package,
            env=environment,
            check=True,
        )
    print(f"All {len(PACKAGES)} immutable package manifests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
