"""Deterministic inspection of the package-local CUDA 11 runtime."""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
from typing import Callable

REQUIRED_SONAMES = (
    "libcudart.so.11.0", "libcublas.so.11", "libcusparse.so.11",
    "libcusolver.so.11",
)
NVIDIA_LIBRARY_COMPONENTS = (
    "cuda_runtime", "cublas", "cusolver", "cusparse",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_library_dirs(prefix: Path, python_version: str = "3.10") -> list[Path]:
    site = prefix / "lib" / f"python{python_version}" / "site-packages" / "nvidia"
    candidates = [prefix / "lib"] + [site / name / "lib" for name in NVIDIA_LIBRARY_COMPONENTS]
    seen: set[str] = set()
    result: list[Path] = []
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if path.is_dir() and key not in seen:
            seen.add(key)
            result.append(path)
    return result


def locate_exact_soname(soname: str, search_dirs: list[Path]) -> Path | None:
    for directory in search_dirs:
        candidate = directory / soname
        if candidate.is_file():
            return candidate.resolve()
    return None


def inspect_runtime(prefix: Path, *, loader: Callable[[str], object] = ctypes.CDLL,
                    python_version: str = "3.10") -> dict:
    prefix = prefix.resolve()
    search_dirs = candidate_library_dirs(prefix, python_version)
    libraries, errors = [], []
    for soname in REQUIRED_SONAMES:
        path = locate_exact_soname(soname, search_dirs)
        record = {"requested_soname": soname, "resolved_path": str(path) if path else None,
                  "loader_succeeded": False, "loader_error": None, "sha256": None}
        if path is None:
            record["loader_error"] = "exact SONAME absent from certified-prefix search directories"
            errors.append(f"{soname}: {record['loader_error']}")
        else:
            try:
                # Load by the exact SONAME DGL requests. The shared activation helper
                # has already placed the certified-prefix directory first.
                loader(soname)
                record["loader_succeeded"] = True
                record["sha256"] = sha256_file(path)
            except OSError as error:
                record["loader_error"] = f"OSError: {error}"
                errors.append(f"{soname}: {record['loader_error']}")
        libraries.append(record)
    return {"environment_prefix": str(prefix), "search_directories": [str(p) for p in search_dirs],
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""), "libraries": libraries,
            "passed": not errors, "errors": errors}
