"""Fail-closed validation of the CPython-3.10 Linux wheel closure."""

from __future__ import annotations

import re

try:  # Available in the package tests and after the locked runtime install.
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
except ImportError:  # The exact pip bootstrap vendors the same standards parser.
    from pip._vendor.packaging.markers import default_environment
    from pip._vendor.packaging.requirements import Requirement

REQUIRED = {
    "torch", "dgl", "nvidia-cuda-runtime-cu11", "nvidia-cublas-cu11",
    "nvidia-cusparse-cu11", "nvidia-cusolver-cu11", "triton", "cmake", "lit",
    "exceptiongroup", "tomli", "pip", "setuptools", "wheel",
}


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def target_environment(evidence: dict) -> dict[str, str]:
    target = evidence.get("target", {})
    environment = default_environment()
    environment.update({
        "implementation_name": "cpython",
        "implementation_version": "3.10.20",
        "os_name": "posix",
        "platform_machine": "x86_64",
        "platform_python_implementation": "CPython",
        "platform_system": "Linux",
        "python_full_version": "3.10.20",
        "python_version": "3.10",
        "sys_platform": "linux",
        "extra": "",
    })
    if target.get("python_full_version") != "3.10.20":
        environment["_invalid_target"] = "true"
    return environment


def parse_locks(lock_texts: list[str]) -> tuple[dict[str, tuple[str, set[str]]], list[str]]:
    locked: dict[str, tuple[str, set[str]]] = {}
    errors: list[str] = []
    for text in lock_texts:
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            direct = re.fullmatch(
                r"([^\s]+)\s+@\s+\S+\s+--hash=sha256:([0-9a-f]{64})", line)
            pinned = re.fullmatch(
                r"([^\s=@]+)==([^\s]+)((?:\s+--hash=sha256:[0-9a-f]{64})+)", line)
            if direct:
                name, version, hashes = normalize(direct.group(1)), None, {direct.group(2)}
            elif pinned:
                name, version = normalize(pinned.group(1)), pinned.group(2)
                hashes = set(re.findall(r"--hash=sha256:([0-9a-f]{64})", pinned.group(3)))
            else:
                errors.append(f"unpinned or unhashed lock line: {line}")
                continue
            if not hashes:
                errors.append(f"lock entry has no accepted SHA-256: {name}")
                continue
            prior = locked.get(name)
            if prior and (version is not None and prior[0] != version):
                errors.append(f"conflicting locked versions for {name}")
            elif prior:
                locked[name] = (prior[0], prior[1] | hashes)
            else:
                locked[name] = (version or "direct-url", hashes)
    return locked, errors


def validate_resolution(evidence: dict, lock_texts: list[str]) -> dict:
    errors: list[str] = []
    if evidence.get("status") != "passed_static_resolution":
        errors.append("static dependency-resolution evidence is not passed")
    target = evidence.get("target", {})
    if target.get("python_full_version") != "3.10.20" or target.get("pip_version") != "25.3":
        errors.append("target must bind CPython 3.10.20 and pip 25.3")
    if target.get("platform_system") != "Linux" or target.get("platform_machine") != "x86_64":
        errors.append("target must bind Linux x86_64 marker semantics")

    rows = [row for row in evidence.get("wheels", []) if row.get("selected_for_lock")]
    selected = {normalize(row["normalized_name"]): row for row in rows}
    if len(selected) != len(rows):
        errors.append("duplicate selected distribution records")
    if "nvidia-nvjitlink-cu11" in selected or evidence.get("invalid_nvjitlink_requirement_present") is not False:
        errors.append("invalid nonexistent CUDA-11 nvJitLink distribution present")
    missing = sorted(REQUIRED - set(selected))
    if missing:
        errors.append(f"unavailable locked distributions: {missing}")

    locked, lock_errors = parse_locks(lock_texts)
    errors.extend(lock_errors)
    unlocked = sorted(set(selected) - set(locked))
    undeclared = sorted(set(locked) - set(selected))
    if unlocked:
        errors.append(f"selected distributions absent from locks: {unlocked}")
    if undeclared:
        errors.append(f"locked distributions absent from evidence: {undeclared}")

    for name, row in selected.items():
        entry = locked.get(name)
        if not entry:
            continue
        version, hashes = entry
        if version != "direct-url" and row["version"] != version:
            errors.append(f"locked artifact unavailable/version mismatch: {name}=={version}")
        if row["sha256"] not in hashes:
            errors.append(f"locked artifact unavailable or hash mismatch: {name}")

    environment = target_environment(evidence)
    active_graph: dict[str, set[str]] = {name: set() for name in selected}
    for owner, row in selected.items():
        for raw in row.get("requires_dist") or []:
            try:
                requirement = Requirement(raw)
                active = requirement.marker is None or requirement.marker.evaluate(environment)
            except Exception as error:
                errors.append(f"invalid Requires-Dist for {owner}: {raw}: {error}")
                continue
            if not active:
                continue
            dependency = normalize(requirement.name)
            active_graph[owner].add(dependency)
            child = selected.get(dependency)
            if child is None:
                errors.append(f"active target dependency is unpinned: {owner} requires {raw}")
            elif child["version"] not in requirement.specifier:
                errors.append(
                    f"active target dependency version mismatch: {owner} requires {raw}; "
                    f"locked {child['version']}")

    roots = {normalize(name) for name in evidence.get("root_distributions", [])}
    missing_roots = sorted(roots - set(selected))
    if missing_roots:
        errors.append(f"target root distributions are absent: {missing_roots}")
    reachable: set[str] = set()
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        pending.extend(active_graph.get(name, set()) - reachable)
    orphaned = sorted(set(selected) - reachable)
    if orphaned:
        errors.append(f"selected distributions are not in the target closure: {orphaned}")

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "selected_distribution_count": len(selected),
        "target_marker_environment": {key: environment[key] for key in (
            "python_full_version", "platform_system", "platform_machine", "sys_platform")},
    }
