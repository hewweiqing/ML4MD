"""Safe orchestration of DelftBlue sbatch --test-only simulations."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .slurm_resources import audit_slurm_resources

FORBIDDEN_OUTPUT = ("error:", "allocation failure", "per-node basis",
    "does not request complete nodes", "specify '--ntasks' and '--cpus-per-task'")


def run_test_only_validation(slurm_dir: Path, *, runner=subprocess.run, sbatch_executable: str | None = None) -> list[dict]:
    slurm_dir = Path(slurm_dir)
    audit_slurm_resources(slurm_dir)
    executable = sbatch_executable or shutil.which("sbatch")
    if not executable:
        raise RuntimeError("sbatch is unavailable; no test-only simulations were run")
    records = []
    for path in sorted(slurm_dir.glob("*.sbatch")):
        completed = runner([executable, "--test-only", str(path)], text=True, capture_output=True, check=False)
        stdout, stderr = completed.stdout or "", completed.stderr or ""
        combined = f"{stdout}\n{stderr}".lower()
        forbidden = [token for token in FORBIDDEN_OUTPUT if token in combined]
        if completed.returncode != 0 or forbidden:
            raise RuntimeError(f"DelftBlue test-only simulation failed for {path.name}: "
                f"returncode={completed.returncode} forbidden={forbidden} stdout={stdout!r} stderr={stderr!r}")
        records.append({"file": path.name, "returncode": completed.returncode,
            "stdout": stdout.strip(), "stderr": stderr.strip(), "simulation_only": True})
    return records
