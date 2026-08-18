"""Parse and enforce DelftBlue Slurm resource headers for every job."""
from __future__ import annotations

import re
from pathlib import Path

_MEMORY = re.compile(r"^([0-9]+(?:\.[0-9]+)?)([A-Za-z]*)$")
_MULTIPLIERS_MB = {"": 1.0, "m": 1.0, "mb": 1.0, "mib": 1.0,
    "g": 1024.0, "gb": 1024.0, "gib": 1024.0,
    "t": 1024.0 * 1024.0, "tb": 1024.0 * 1024.0, "tib": 1024.0 * 1024.0}
PARTITION_LIMITS_MB_PER_CPU = {"gpu-a100": 8000.0, "gpu-a100-small": 8000.0,
    "compute": 3968.0, "compute-p1": 3968.0}


def memory_megabytes(value: str) -> float:
    match = _MEMORY.fullmatch(value.strip())
    if not match:
        raise ValueError(f"invalid Slurm memory value: {value!r}")
    unit = match.group(2).lower()
    if unit not in _MULTIPLIERS_MB:
        raise ValueError(f"unsupported Slurm memory unit: {match.group(2)!r}")
    result = float(match.group(1)) * _MULTIPLIERS_MB[unit]
    if result <= 0:
        raise ValueError("Slurm memory must be positive")
    return result


def parse_sbatch_headers(path: Path) -> dict[str, str]:
    headers = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = re.match(r"^#SBATCH\s+--([^=\s]+)(?:(?:=|\s+)(\S+))?\s*$", line)
        if match:
            headers[match.group(1)] = match.group(2) or "true"
    return headers


def _positive_integer(headers: dict[str, str], name: str, filename: str) -> int:
    try:
        value = int(headers[name])
    except (KeyError, ValueError) as error:
        raise RuntimeError(f"{filename} requires a positive --{name}") from error
    if value <= 0:
        raise RuntimeError(f"{filename} requires a positive --{name}")
    return value


def audit_slurm_resources(slurm_dir: Path) -> list[dict]:
    paths = sorted(Path(slurm_dir).glob("*.sbatch"))
    if not paths:
        raise RuntimeError("no Slurm scripts were audited")
    records = []
    for path in paths:
        headers = parse_sbatch_headers(path)
        partition = headers.get("partition")
        if partition not in PARTITION_LIMITS_MB_PER_CPU:
            raise RuntimeError(f"{path.name} has unsupported or missing partition: {partition}")
        ntasks = _positive_integer(headers, "ntasks", path.name)
        cpus = _positive_integer(headers, "cpus-per-task", path.name)
        if "mem-per-cpu" not in headers:
            raise RuntimeError(f"{path.name} lacks --mem-per-cpu")
        normalized = memory_megabytes(headers["mem-per-cpu"])
        limit = PARTITION_LIMITS_MB_PER_CPU[partition]
        if normalized > limit:
            raise RuntimeError(f"{path.name} requests {normalized:g} MB per CPU on {partition}; DelftBlue maximum is {limit:g} MB")
        if "nodes" in headers and "exclusive" not in headers and ntasks == 1:
            raise RuntimeError(f"{path.name} uses unnecessary --nodes on a non-exclusive single-task job")
        records.append({"file": path.name, "partition": partition,
            "effective_partition_policy": "compute-p1" if partition == "compute" else partition,
            "declared_memory_per_cpu": headers["mem-per-cpu"], "declared": headers["mem-per-cpu"],
            "normalized_mb_per_cpu": normalized, "maximum_mb_per_cpu": limit,
            "ntasks": ntasks, "cpus_per_task": cpus, "nodes_declared": "nodes" in headers,
            "exclusive": "exclusive" in headers})
    return records


def audit_gpu_a100_memory(slurm_dir: Path, maximum_mb_per_cpu: float = 8000.0) -> list[dict]:
    """Backward-compatible filtered view used by inherited v12 tests."""
    if maximum_mb_per_cpu != PARTITION_LIMITS_MB_PER_CPU["gpu-a100"]:
        raise ValueError("gpu-a100 maximum is frozen at 8000 MB per CPU")
    return [row for row in audit_slurm_resources(slurm_dir) if row["partition"] == "gpu-a100"]
