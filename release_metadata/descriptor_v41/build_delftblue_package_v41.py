from __future__ import annotations

import ast
import gzip
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "delftblue_package_v41"
ARCHIVE = ROOT / "alignn_stage2_delftblue_v41.tar.gz"
SIDECAR = ROOT / "DELFTBLUE_ARCHIVE_MANIFEST_V41.json"
V40_ARCHIVE_SHA256 = "ec3701ffa164e28026deb4e76d4c77a518b5a7c635ee8335db470f8c9959d99e"
V40_AGGREGATE = "7fe3255b5f688b4c952f34181ebb5bed35b87609b253bd5bf9215ee2daeaa795"
V26_ARCHIVE_SHA256 = "f4781d8946145202f22633a6631283335f7fb012490a472002335927ea5f8996"
V26_AGGREGATE = "1c15f877cee4eb6eebfd096962c6a793d0d3f14385287a6a9385e3adce05bd7d"
V26_MUBEN = "868654151b87d526f122f741183c23599913541d5d62aada5cb35c91bc510719"
V31_ARCHIVE_SHA256 = "c4f36085a7e24908822b81d7224f9c20780a89aefe8096f44760188909589d0e"
V32_ARCHIVE_SHA256 = "7df66a032c7bd2070fb1ff5f1f5f3993c37f1f3ee9e2ccfd461ebb110ed5539c"
V33_ARCHIVE_SHA256 = "6b8395256d54d7ef41189539bc301c4e7e69f13757f27f3b8696b2bb8d80d3fc"
V34_ARCHIVE_SHA256 = "e7f689e5d0bb4b22e1af0b869a93c3a65e52a377ff030c3319e175419db3c007"
V34_AGGREGATE = "62752e9e117da6935372e0d2a40924a2a48809d9b0144c98b775910d34cc2629"
V35_ARCHIVE_SHA256 = "4f5efb958bafd0ed4b554c7dea064768d3af7796a04625826c3c4c66486cb726"
V35_AGGREGATE = "827cd13dc3583664f16d607ac444f08802f6e44564deee812815a294ca79e550"
V36_MUBEN = "108b3183400725ca53e025c6f0f92690f35069f6fd786091b9d37dc50e90f08b"
V36_ARCHIVE_SHA256 = "77b4525872945e5d18fbb197bc7628371d46a572555b2b8b200869593aaa5bb2"
V36_AGGREGATE = "f28267cb8d6d4f094d33bee125c1fbf69dc908bfa637498f38129c8b8f513c38"
V37_ARCHIVE_SHA256 = "57790c9fe88f9c1312302340c7014f54ba9e87f9092543180aa5569205b139d0"
V37_AGGREGATE = "d218cdb429ee4b0e803ecee6ef3c739252b394bda3924677b7dff96c3d203065"
V38_ARCHIVE_SHA256 = "08ed1d13070d079d50b58dec6b2786bc380a5bc4023266efb4969b4726ea43d1"
V38_AGGREGATE = "eecf323e84d99f91145ab11b5017870edf34fef6a7fe537a873ca53289b4549b"
V39_ARCHIVE_SHA256 = "0016e6fb24384e992d86ebc1d94a27582707441f5a4db40e42e44d1b81c43221"
V39_AGGREGATE = "914e610be6b857bbb54b699bcfeb4e425b47529b87cce63793df644125d51ca1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(root: Path) -> None:
    generated_dirs = {"__pycache__", ".pytest_cache", "logs", "preflight", "outputs",
        "runtime_outputs", "results", "checkpoints", "graph_cache", "environment_reports"}
    for path in sorted(Path(root).rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir() and path.name in generated_dirs:
            shutil.rmtree(path)
        elif path.is_file() and path.suffix.lower() in {".pyc", ".pyo", ".tmp", ".part"}:
            path.unlink()


def normalized(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    info.mtime = 0
    executable = info.isdir() or info.name.endswith((".sh", ".sbatch")) or (
        "/scripts/" in info.name and info.name.endswith(".py"))
    info.mode = 0o755 if executable else 0o644
    return info


def bash_executable() -> str:
    for candidate in (r"C:\Program Files\Git\bin\bash.exe", r"C:\msys64\usr\bin\bash.exe", shutil.which("bash")):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError("bash is required for shell syntax checks")


def run_pure_behavioral_tests(root: Path) -> list[str]:
    path = root / "tests/test_v31_recovery.py"
    spec = importlib.util.spec_from_file_location("v36_behavioral_tests", path)
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    tests = sorted(name for name in vars(module) if name.startswith("test_"))
    for name in tests:
        getattr(module, name)()
    from alignn_stage2.production import assert_temperature_invariance
    raw_margin = np.array([1.0e-5, 1.005e-5, 1.0, 2.0])
    scaled_margin = np.array([0.5002e-5, 0.5001e-5, 0.5, 1.0])
    raw = np.column_stack((np.zeros_like(raw_margin), raw_margin))
    scaled = np.column_stack((np.zeros_like(scaled_margin), scaled_margin))
    result = assert_temperature_invariance(raw, scaled)
    if not result["ranking_unchanged"] or not result["tie_pattern_unchanged"]:
        raise RuntimeError("roundoff-scale production-ordering regression")
    inverted = np.column_stack((np.zeros(3), np.array([0.2, 1.4, 0.8])))
    ordered = np.column_stack((np.zeros(3), np.array([0.2, 0.8, 1.4])))
    try:
        assert_temperature_invariance(ordered, inverted)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("material production-ordering inversion was accepted")
    v38_path = root / "tests/test_v38_full_grid_recovery.py"
    v38_spec = importlib.util.spec_from_file_location("v38_behavioral_tests", v38_path)
    v38_module = importlib.util.module_from_spec(v38_spec)
    v38_spec.loader.exec_module(v38_module)
    v38_tests = sorted(name for name in vars(v38_module) if name.startswith("test_"))
    for name in v38_tests:
        getattr(v38_module, name)()
    v40_path = root / "tests/test_v40_numerical_resolution.py"
    v40_spec = importlib.util.spec_from_file_location("v40_behavioral_tests", v40_path)
    v40_module = importlib.util.module_from_spec(v40_spec)
    v40_spec.loader.exec_module(v40_module)
    v40_tests = [
        "test_authorization_is_exact_and_honestly_post_training_pre_outer",
        "test_assessment_accepts_authorized_stationarity_but_preserves_false_convergence",
        "test_assessment_fails_closed_on_each_required_condition",
        "test_fitted_object_preserves_original_nonconvergence_metadata",
        "test_v40_validator_is_validation_only_and_export_order_is_fail_closed",
    ]
    for name in v40_tests:
        getattr(v40_module, name)()
    v41_path = root / "tests/test_v41_oof_evidence_binding.py"
    v41_spec = importlib.util.spec_from_file_location("v41_behavioral_tests", v41_path)
    v41_module = importlib.util.module_from_spec(v41_spec)
    v41_spec.loader.exec_module(v41_module)
    v41_tests = sorted(name for name in vars(v41_module) if name.startswith("test_"))
    for name in v41_tests:
        getattr(v41_module, name)()
    return [*tests, *v38_tests, *v40_tests, *v41_tests,
        "production_roundoff_tie_accepted", "production_material_inversion_rejected"]


def static_checks(root: Path) -> dict:
    python_files = sorted(root.rglob("*.py"))
    json_files = sorted(root.rglob("*.json"))
    shell_files = sorted([*root.rglob("*.sh"), *root.rglob("*.sbatch")])
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))
    bash = bash_executable()
    for path in shell_files:
        result = subprocess.run([bash, "-n", str(path)], text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError({"bash_n": str(path), "stderr": result.stderr})
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    resource = subprocess.run([sys.executable, str(root / "scripts/audit_slurm_resources.py")],
        cwd=root, env=env, text=True, capture_output=True, check=False)
    if resource.returncode:
        raise RuntimeError({"resource_audit_stdout": resource.stdout, "resource_audit_stderr": resource.stderr})
    resource_value = json.loads(resource.stdout)
    return {"status": "passed", "python_ast_files": len(python_files),
        "json_files": len(json_files), "bash_syntax_files": len(shell_files),
        "resource_audited_jobs": resource_value["audited_job_count"],
        "behavioral_tests": run_pure_behavioral_tests(root)}


def archive_audit(path: Path) -> dict:
    with tarfile.open(path, "r:gz") as handle:
        members = handle.getmembers()
    names = [member.name for member in members]
    failures = []
    if len(names) != len(set(names)):
        failures.append("duplicate members")
    for member in members:
        pure = PurePosixPath(member.name)
        if pure.is_absolute() or ".." in pure.parts or "\\" in member.name:
            failures.append(f"unsafe path: {member.name}")
        executable = member.isdir() or member.name.endswith((".sh", ".sbatch")) or (
            "/scripts/" in member.name and member.name.endswith(".py"))
        expected = 0o755 if executable else 0o644
        if member.mode != expected:
            failures.append(f"permission mismatch: {member.name}")
    if failures:
        raise RuntimeError(failures)
    return {"status": "passed", "member_count": len(members), "duplicate_members": [],
        "unsafe_paths": [], "permission_mismatches": []}


def make_archive() -> None:
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False, dir=ROOT) as temporary:
        tar_path = Path(temporary.name)
    try:
        with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as handle:
            handle.add(PACKAGE, arcname=PACKAGE.name, filter=normalized)
        with tar_path.open("rb") as source, ARCHIVE.open("wb") as destination:
            with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0, compresslevel=9) as zipped:
                shutil.copyfileobj(source, zipped)
    finally:
        tar_path.unlink(missing_ok=True)


def main() -> int:
    if ARCHIVE.exists() or SIDECAR.exists():
        raise SystemExit("refusing to overwrite v41 archive or sidecar")
    if sha256(ROOT / "alignn_stage2_delftblue_v40.tar.gz") != V40_ARCHIVE_SHA256:
        raise RuntimeError("immutable v40 archive mismatch")
    v40_manifest = json.loads((ROOT / "delftblue_package_v40/PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    if v40_manifest.get("aggregate_sha256") != V40_AGGREGATE:
        raise RuntimeError("immutable v40 package aggregate mismatch")
    v26_archive = ROOT / "alignn_stage2_delftblue_v26.tar.gz"
    if sha256(v26_archive) != V26_ARCHIVE_SHA256:
        raise RuntimeError("immutable v26 archive mismatch")
    if sha256(ROOT / "alignn_stage2_delftblue_v31.tar.gz") != V31_ARCHIVE_SHA256:
        raise RuntimeError("immutable v31 archive mismatch")
    if sha256(ROOT / "alignn_stage2_delftblue_v32.tar.gz") != V32_ARCHIVE_SHA256:
        raise RuntimeError("immutable v32 archive mismatch")
    if sha256(ROOT / "alignn_stage2_delftblue_v33.tar.gz") != V33_ARCHIVE_SHA256:
        raise RuntimeError("immutable v33 archive mismatch")
    if sha256(ROOT / "alignn_stage2_delftblue_v34.tar.gz") != V34_ARCHIVE_SHA256:
        raise RuntimeError("immutable v34 archive mismatch")
    v34_manifest = json.loads((ROOT / "delftblue_package_v34/PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    if v34_manifest.get("aggregate_sha256") != V34_AGGREGATE:
        raise RuntimeError("immutable v34 package aggregate mismatch")
    if sha256(ROOT / "alignn_stage2_delftblue_v35.tar.gz") != V35_ARCHIVE_SHA256:
        raise RuntimeError("immutable v35 archive mismatch")
    v35_manifest = json.loads((ROOT / "delftblue_package_v35/PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    if v35_manifest.get("aggregate_sha256") != V35_AGGREGATE:
        raise RuntimeError("immutable v35 package aggregate mismatch")
    if sha256(ROOT / "alignn_stage2_delftblue_v36.tar.gz") != V36_ARCHIVE_SHA256:
        raise RuntimeError("immutable v36 archive mismatch")
    v36_manifest = json.loads((ROOT / "delftblue_package_v36/PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    if v36_manifest.get("aggregate_sha256") != V36_AGGREGATE:
        raise RuntimeError("immutable v36 package aggregate mismatch")
    if sha256(ROOT / "alignn_stage2_delftblue_v37.tar.gz") != V37_ARCHIVE_SHA256:
        raise RuntimeError("immutable v37 archive mismatch")
    v37_manifest = json.loads((ROOT / "delftblue_package_v37/PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    if v37_manifest.get("aggregate_sha256") != V37_AGGREGATE:
        raise RuntimeError("immutable v37 package aggregate mismatch")
    if sha256(ROOT / "alignn_stage2_delftblue_v38.tar.gz") != V38_ARCHIVE_SHA256:
        raise RuntimeError("immutable v38 archive mismatch")
    v38_manifest = json.loads((ROOT / "delftblue_package_v38/PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    if v38_manifest.get("aggregate_sha256") != V38_AGGREGATE:
        raise RuntimeError("immutable v38 package aggregate mismatch")
    if sha256(ROOT / "alignn_stage2_delftblue_v39.tar.gz") != V39_ARCHIVE_SHA256:
        raise RuntimeError("immutable v39 archive mismatch")
    v39_manifest = json.loads((ROOT / "delftblue_package_v39/PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    if v39_manifest.get("aggregate_sha256") != V39_AGGREGATE:
        raise RuntimeError("immutable v39 package aggregate mismatch")
    v26_manifest = json.loads((ROOT / "delftblue_package_v26/PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    if v26_manifest.get("aggregate_sha256") != V26_AGGREGATE:
        raise RuntimeError("immutable v26 package aggregate mismatch")
    if sha256(ROOT / "delftblue_package_v26/vendor/muben_temperature_scaling.py") != V26_MUBEN:
        raise RuntimeError("immutable v26 MUBen source mismatch")
    if sha256(PACKAGE / "vendor/muben_temperature_scaling.py") != V36_MUBEN:
        raise RuntimeError("v40 MUBen source does not match the amendment approval")
    older_before = {path.name: sha256(path) for path in ROOT.glob("alignn_stage2_delftblue*.tar.gz")}

    clean(PACKAGE)
    runbook = PACKAGE / "V41_OOF_RUNBOOK.md"
    runbook.write_text(runbook.read_text(encoding="utf-8").replace(
        "V41_ARCHIVE_SHA256_PLACEHOLDER",
        "$(python3 -c 'import json; print(json.load(open(\"DELFTBLUE_ARCHIVE_MANIFEST_V41.json\"))[\"archive_sha256\"])')"),
        encoding="utf-8", newline="\n")
    manifest_path = PACKAGE / "PACKAGE_MANIFEST.json"
    manifest_path.unlink(missing_ok=True)
    sys.path.insert(0, str(PACKAGE))
    from alignn_stage2.common import package_manifest, verify_manifest, write_json
    manifest = package_manifest(PACKAGE)
    write_json(manifest_path, manifest)
    verification = verify_manifest(PACKAGE)
    if verification["status"] != "passed":
        raise RuntimeError(verification)
    source_checks = static_checks(PACKAGE)
    clean(PACKAGE)
    make_archive()
    audit = archive_audit(ARCHIVE)

    with tempfile.TemporaryDirectory(prefix="v41_extract_", dir=ROOT) as temporary:
        with tarfile.open(ARCHIVE, "r:gz") as handle:
            handle.extractall(temporary)
        extracted = Path(temporary) / PACKAGE.name
        env = os.environ.copy()
        env["PYTHONPATH"] = str(extracted)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        verify = subprocess.run([sys.executable, str(extracted / "scripts/verify_package.py")],
            cwd=extracted, env=env, text=True, capture_output=True, check=False)
        if verify.returncode:
            raise RuntimeError({"clean_verify_stdout": verify.stdout, "clean_verify_stderr": verify.stderr})
        clean_checks = static_checks(extracted)

    older_after = {path.name: sha256(path) for path in ROOT.glob("alignn_stage2_delftblue*.tar.gz") if path != ARCHIVE}
    if older_before != older_after:
        raise RuntimeError("an older archive changed")
    sidecar = {
        "schema_version": 1,
        "release": "delftblue_package_v41",
        "archive": ARCHIVE.name,
        "archive_sha256": sha256(ARCHIVE),
        "archive_size_bytes": ARCHIVE.stat().st_size,
        "package_aggregate_sha256": manifest["aggregate_sha256"],
        "source_package_verification": verification,
        "source_checks": source_checks,
        "clean_extraction_checks": clean_checks,
        "archive_audit": audit,
        "preserved_v26_archive_sha256": V26_ARCHIVE_SHA256,
        "preserved_v26_package_aggregate_sha256": V26_AGGREGATE,
        "preserved_v26_muben_sha256": V26_MUBEN,
        "preserved_v31_archive_sha256": V31_ARCHIVE_SHA256,
        "preserved_v32_archive_sha256": V32_ARCHIVE_SHA256,
        "preserved_v33_archive_sha256": V33_ARCHIVE_SHA256,
        "preserved_v34_archive_sha256": V34_ARCHIVE_SHA256,
        "preserved_v34_package_aggregate_sha256": V34_AGGREGATE,
        "preserved_v35_archive_sha256": V35_ARCHIVE_SHA256,
        "preserved_v35_package_aggregate_sha256": V35_AGGREGATE,
        "preserved_v36_archive_sha256": V36_ARCHIVE_SHA256,
        "preserved_v36_package_aggregate_sha256": V36_AGGREGATE,
        "preserved_v37_archive_sha256": V37_ARCHIVE_SHA256,
        "preserved_v37_package_aggregate_sha256": V37_AGGREGATE,
        "preserved_v38_archive_sha256": V38_ARCHIVE_SHA256,
        "preserved_v38_package_aggregate_sha256": V38_AGGREGATE,
        "preserved_v39_archive_sha256": V39_ARCHIVE_SHA256,
        "preserved_v39_package_aggregate_sha256": V39_AGGREGATE,
        "preserved_v40_archive_sha256": V40_ARCHIVE_SHA256,
        "preserved_v40_package_aggregate_sha256": V40_AGGREGATE,
        "v40_muben_sha256": V36_MUBEN,
        "preserved_older_archive_sha256": older_after,
        "training_calibration_inference": "not_run",
        "outer_test_accessed": False,
        "pip_network_slurm_cluster_operations": "not_run",
        "scientific_change": "none; operational OOF/final-audit evidence paths corrected to immutable v26/v36 external gates plus full-grid hash audit",
    }
    SIDECAR.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"archive": str(ARCHIVE), "archive_sha256": sidecar["archive_sha256"],
        "package_aggregate_sha256": manifest["aggregate_sha256"], "members": audit["member_count"],
        "source_checks": source_checks, "clean_extraction_checks": clean_checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



