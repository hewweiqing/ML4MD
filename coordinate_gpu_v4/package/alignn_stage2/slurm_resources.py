"""Static DelftBlue resource-policy audit for every package Slurm script."""
from __future__ import annotations

import re
from pathlib import Path


def _headers(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"#SBATCH\s+--([^=\s]+)(?:=(\S+)|\s+(\S+))?", line)
        if match:
            result[match.group(1)] = match.group(2) or match.group(3) or "true"
    return result


def _memory_mb(value: str) -> int:
    match = re.fullmatch(r"(?i)(\d+)(M|MB|MiB|G|GB|GiB)", value)
    if not match:
        raise RuntimeError(f"unsupported Slurm memory unit: {value}")
    amount, unit = int(match.group(1)), match.group(2).lower()
    return amount if unit in {"m", "mb", "mib"} else amount * 1024


def audit_slurm_resources(slurm_dir: Path) -> list[dict]:
    records = []
    for path in sorted(Path(slurm_dir).glob("*.sbatch")):
        headers = _headers(path)
        required = {"account", "partition", "ntasks", "cpus-per-task", "mem-per-cpu", "time"}
        missing = sorted(required - headers.keys())
        if missing:
            raise RuntimeError(f"{path.name}: missing directives {missing}")
        if headers["account"] != "research-ME-mse" or headers["ntasks"] != "1":
            raise RuntimeError(f"{path.name}: invalid account/task topology")
        if "nodes" in headers and "exclusive" not in headers:
            raise RuntimeError(f"{path.name}: unnecessary --nodes on non-exclusive single-task job")
        partition = headers["partition"]
        memory = _memory_mb(headers["mem-per-cpu"])
        limit = 8000 if partition.startswith("gpu-a100") else 3968 if partition == "compute" else None
        if limit is None or memory > limit:
            raise RuntimeError(f"{path.name}: partition/memory policy violation")
        cpus = int(headers["cpus-per-task"])
        expected_cpus = 2 if partition == "gpu-a100-small" else 8
        if cpus != expected_cpus:
            raise RuntimeError(f"{path.name}: expected {expected_cpus} CPUs for {partition}")
        gpu = partition.startswith("gpu-a100")
        if gpu != (headers.get("gpus-per-task") == "1"):
            raise RuntimeError(f"{path.name}: GPU directive/partition mismatch")
        records.append({"file": path.name, "partition": partition, "cpus_per_task": cpus,
            "normalized_mb_per_cpu": memory, "gpus_per_task": int(headers.get("gpus-per-task", "0"))})
    if not records:
        raise RuntimeError("no Slurm scripts found")
    return records
