from __future__ import annotations
import ast, gzip, hashlib, json, os, shutil, subprocess, sys, tarfile, tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "delftblue_coordinate_gpu_v4"
ARCHIVE = ROOT / "alignn_coordinate_gpu_delftblue_v4.tar.gz"
SIDECAR = ROOT / "DELFTBLUE_COORDINATE_GPU_ARCHIVE_MANIFEST_V4.json"
MUBEN = "108b3183400725ca53e025c6f0f92690f35069f6fd786091b9d37dc50e90f08b"
PRESERVE = {
    "alignn_coordinate_gpu_delftblue_v3.tar.gz": "7da53c3fa34742c9bc78686c869cfea69c10e9a33f4fb1987afa0d7f23ba1231",
    "alignn_coordinate_gpu_delftblue_v2.tar.gz": "971d1b6c97532e8f9f45ab9620ff9e7f14168860a2fb072eca9c3f59cc8800c5",
    "alignn_coordinate_gpu_delftblue_v1.tar.gz": "69cd767aea2a28c38f6d7131281ca9fd45a517c87ea232702954fdc65d8f0414",
    "alignn_stage2_delftblue_v29.tar.gz": "721c7128946287ba7a6db75934618c26e4e204a602aa2f9ed6f7ce4a14625e95",
    "alignn_stage2_delftblue_v30.tar.gz": "9145c799b433532da2ac359edc06e3ba29221256a11f5c29e1a41047a0803722",
    "alignn_stage2_delftblue_v36.tar.gz": "77b4525872945e5d18fbb197bc7628371d46a572555b2b8b200869593aaa5bb2",
    "alignn_stage2_delftblue_v37.tar.gz": "57790c9fe88f9c1312302340c7014f54ba9e87f9092543180aa5569205b139d0",
}
GENERATED = {"__pycache__", ".pytest_cache", "logs", "preflight", "runtime_outputs", "outputs", "results", "checkpoints", "graph_cache", "environment_reports"}

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

def clean(root):
    for path in sorted(Path(root).rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir() and path.name in GENERATED: shutil.rmtree(path)
        elif path.is_file() and path.suffix.lower() in {".pyc", ".pyo", ".tmp", ".part"}: path.unlink()

def bash_executable():
    for item in (r"C:\Program Files\Git\bin\bash.exe", r"C:\msys64\usr\bin\bash.exe", shutil.which("bash")):
        if item and Path(item).is_file(): return str(item)
    raise RuntimeError("bash is required")

def run_checks(root):
    root = Path(root); py = list(root.rglob("*.py")); js = list(root.rglob("*.json")); shell = [*root.rglob("*.sh"), *root.rglob("*.sbatch")]
    for path in py: ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for path in js: json.loads(path.read_text(encoding="utf-8"))
    bash = bash_executable()
    for path in shell:
        done = subprocess.run([bash, "-n", str(path)], text=True, capture_output=True, check=False)
        if done.returncode: raise RuntimeError({"bash_n": str(path), "stderr": done.stderr})
    env = os.environ.copy(); env.update({"PYTHONPATH": str(root) + os.pathsep + env.get("PYTHONPATH", ""), "PYTHONDONTWRITEBYTECODE": "1"})
    outputs = []
    for command in ([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests_gpu"], [sys.executable, "scripts/audit_gpu_coordinate_resources.py"]):
        done = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, check=False)
        if done.returncode: raise RuntimeError({"command": command, "stdout": done.stdout, "stderr": done.stderr})
        outputs.append({"command": command, "returncode": 0, "stdout": done.stdout.strip()})
    return {"status": "passed", "python_ast_files": len(py), "json_files": len(js), "bash_syntax_files": len(shell), "behavioral_checks": outputs}

def normalized(info):
    info.uid = info.gid = 0; info.uname = info.gname = "root"; info.mtime = 0
    executable = info.isdir() or info.name.endswith((".sh", ".sbatch")) or "/scripts/" in info.name and info.name.endswith(".py")
    info.mode = 0o755 if executable else 0o644
    return info

def make_archive():
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False, dir=ROOT) as temporary: tar_path = Path(temporary.name)
    try:
        with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as handle: handle.add(PACKAGE, arcname=PACKAGE.name, filter=normalized)
        with tar_path.open("rb") as source, ARCHIVE.open("wb") as destination:
            with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0, compresslevel=9) as zipped: shutil.copyfileobj(source, zipped)
    finally: tar_path.unlink(missing_ok=True)

def audit_archive(path):
    with tarfile.open(path, "r:gz") as handle: members = handle.getmembers()
    names = [item.name for item in members]; failures = []
    if len(names) != len(set(names)): failures.append("duplicate members")
    for item in members:
        parsed = PurePosixPath(item.name)
        if parsed.is_absolute() or ".." in parsed.parts or "\\" in item.name: failures.append(f"unsafe:{item.name}")
        executable = item.isdir() or item.name.endswith((".sh", ".sbatch")) or "/scripts/" in item.name and item.name.endswith(".py")
        if item.mode != (0o755 if executable else 0o644): failures.append(f"mode:{item.name}")
    if failures: raise RuntimeError(failures)
    return {"status": "passed", "member_count": len(members), "duplicates": [], "unsafe_paths": [], "permission_mismatches": []}

def main():
    if ARCHIVE.exists() or SIDECAR.exists(): raise SystemExit("refusing to overwrite existing GPU-coordinate archive/manifest")
    for name, expected in PRESERVE.items():
        if not (ROOT / name).is_file() or sha(ROOT / name) != expected: raise RuntimeError(f"immutable archive preservation failure: {name}")
    if sha(PACKAGE / "vendor/muben_temperature_scaling.py") != MUBEN: raise RuntimeError("approved MUBen scaler changed")
    before = {name: sha(ROOT / name) for name in PRESERVE}; clean(PACKAGE)
    runbook = PACKAGE / "DELFTBLUE_GPU_COORDINATE_RUNBOOK.md"
    runbook.write_text(runbook.read_text(encoding="utf-8").replace("ARCHIVE_SHA256_PLACEHOLDER", "$(python3 -c 'import json; print(json.load(open(\"DELFTBLUE_COORDINATE_GPU_ARCHIVE_MANIFEST_V4.json\"))[\"archive_sha256\"])')"), encoding="utf-8", newline="\n")
    manifest_path = PACKAGE / "PACKAGE_MANIFEST.json"; manifest_path.unlink(missing_ok=True)
    sys.path.insert(0, str(PACKAGE)); from alignn_stage2.common import package_manifest, verify_manifest, write_json
    manifest = package_manifest(PACKAGE); write_json(manifest_path, manifest); verification = verify_manifest(PACKAGE)
    if verification["status"] != "passed": raise RuntimeError(verification)
    source_checks = run_checks(PACKAGE); clean(PACKAGE); make_archive(); archive_audit = audit_archive(ARCHIVE)
    with tempfile.TemporaryDirectory(prefix="coordinate_gpu_extract_", dir=ROOT) as temporary:
        with tarfile.open(ARCHIVE, "r:gz") as handle: handle.extractall(temporary)
        extracted = Path(temporary) / PACKAGE.name; clean_checks = run_checks(extracted)
        env = os.environ.copy(); env.update({"PYTHONPATH": str(extracted), "PYTHONDONTWRITEBYTECODE": "1"})
        done = subprocess.run([sys.executable, "scripts/verify_package.py"], cwd=extracted, env=env, text=True, capture_output=True, check=False)
        if done.returncode: raise RuntimeError({"clean_manifest": done.stdout, "stderr": done.stderr})
        clean_verification = json.loads(done.stdout)
    after = {name: sha(ROOT / name) for name in PRESERVE}
    if before != after: raise RuntimeError("an older archive changed")
    sidecar = {"schema_version": 1, "release": PACKAGE.name, "archive": ARCHIVE.name, "archive_sha256": sha(ARCHIVE),
        "archive_size_bytes": ARCHIVE.stat().st_size, "package_aggregate_sha256": manifest["aggregate_sha256"],
        "source_package_verification": verification, "source_checks": source_checks, "clean_extraction_checks": clean_checks,
        "clean_extraction_package_verification": clean_verification, "archive_audit": archive_audit,
        "preserved_archives": after, "muben_module_sha256": MUBEN,
        "scientific_change": "none: preserve CPU warm-up permutation stream while transferring selected indices to CUDA", "cpu_checkpoints_imported": False,
        "coordinate_outer_test_results_accessed": False, "training_calibration_inference_pip_network_slurm_operations": "not_run",
        "delftblue_runtime_validation": "pending_required_gates"}
    SIDECAR.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"archive": str(ARCHIVE), "archive_sha256": sidecar["archive_sha256"], "package_aggregate_sha256": manifest["aggregate_sha256"], "members": archive_audit["member_count"], "tests": source_checks["status"], "clean_extraction": clean_checks["status"]}, indent=2))
    return 0

if __name__ == "__main__": raise SystemExit(main())
