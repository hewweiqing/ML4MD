import json
import subprocess
import sys
from pathlib import Path

from alignn_stage2.cuda_runtime import REQUIRED_SONAMES, inspect_runtime

ROOT = Path(__file__).resolve().parents[1]
GPU_JOBS = {"00_a100_preflight.sbatch", "05_build_graph_cache.sbatch",
    "06_smoke_fold0_seed0.sbatch", "07_profile_100_batches.sbatch",
    "08_primary_fold0_seed0.sbatch", "10_train_fold_seed.sbatch", "20_calibrate_export.sbatch"}
COMPUTE_JOBS = {"30_consolidate_oof.sbatch", "40_final_audit.sbatch"}


def make_runtime(prefix: Path, omit: str | None = None) -> Path:
    library_dir = prefix / "lib"
    library_dir.mkdir(parents=True)
    for soname in REQUIRED_SONAMES:
        if soname != omit:
            (library_dir / soname).write_bytes((soname + "\n").encode())
    return library_dir


def accepting_loader(soname: str):
    assert soname in REQUIRED_SONAMES
    return object()


def test_v13_failure_missing_libcusparse_fails(tmp_path):
    make_runtime(tmp_path, "libcusparse.so.11")
    report = inspect_runtime(tmp_path, loader=accepting_loader)
    row = next(item for item in report["libraries"] if item["requested_soname"] == "libcusparse.so.11")
    assert not report["passed"] and not row["loader_succeeded"] and row["resolved_path"] is None


def test_missing_cuda_search_path_fails(tmp_path):
    report = inspect_runtime(tmp_path, loader=accepting_loader)
    assert not report["passed"] and report["search_directories"] == []
    assert len(report["errors"]) == len(REQUIRED_SONAMES)


def test_complete_dependency_path_passes_and_hashes_libraries(tmp_path):
    library_dir = make_runtime(tmp_path)
    report = inspect_runtime(tmp_path, loader=accepting_loader)
    assert report["passed"] and report["search_directories"] == [str(library_dir.resolve())]
    assert all(row["loader_succeeded"] and len(row["sha256"]) == 64 for row in report["libraries"])


def test_loader_failure_is_fail_closed(tmp_path):
    make_runtime(tmp_path)
    def rejecting_loader(path: str):
        if path.endswith("libcusparse.so.11"):
            raise OSError("synthetic missing dependency")
        return object()
    report = inspect_runtime(tmp_path, loader=rejecting_loader)
    assert not report["passed"] and any("synthetic missing dependency" in e for e in report["errors"])


def test_gpu_jobs_use_one_helper_before_python_and_compute_jobs_do_not():
    for name in GPU_JOBS:
        source = (ROOT / "slurm" / name).read_text(encoding="utf-8")
        assert source.count("source scripts/activate_cuda_runtime.sh") == 1
        assert source.index("source scripts/activate_cuda_runtime.sh") < source.index('srun "$ENV_PREFIX/bin/python"')
        assert "cu118_v9" in source and "cu118_v8" not in source and "cu118_v7" not in source and "cu118_v5" not in source
    for name in COMPUTE_JOBS:
        source = (ROOT / "slurm" / name).read_text(encoding="utf-8")
        assert "activate_cuda_runtime.sh" not in source
        assert "cu118_v9" in source and "cu118_v8" not in source and "cu118_v7" not in source and "cu118_v5" not in source


def test_shared_helper_is_fail_closed_and_not_pwd_dependent():
    source = (ROOT / "scripts" / "activate_cuda_runtime.sh").read_text(encoding="utf-8")
    assert '${BASH_SOURCE[0]}' in source and '"$PWD"' not in source
    assert 'LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH' in source
    assert '[[ -d "$_runtime_dir" ]]' in source
    assert '.alignn_stage2_v9_login_certification.json' in source


def test_setup_uses_same_helper_before_certification():
    source = (ROOT / "setup_environment.sh").read_text(encoding="utf-8")
    assert 'source "$PACKAGE_DIR/scripts/activate_cuda_runtime.sh"' in source
    assert source.index('source "$PACKAGE_DIR/scripts/activate_cuda_runtime.sh"') < source.rindex('scripts/verify_environment.py')
    assert 'CUDA11_RUNTIME_REQUIREMENTS.txt' in source


def test_cuda_runtime_components_are_exactly_pinned_and_lock_declares_sonames():
    lines = [line for line in (ROOT / "CUDA11_RUNTIME_REQUIREMENTS.txt").read_text(encoding="utf-8").splitlines()
             if line and not line.startswith("#")]
    assert len(lines) == 4 and all(line.count("==") == 1 and "--hash=sha256:" in line for line in lines)
    assert any(line.startswith("nvidia-cusparse-cu11==") for line in lines)
    assert not any("nvjitlink" in line.lower() for line in lines)
    lock = json.loads((ROOT / "CUDA11_RUNTIME_LOCK.json").read_text(encoding="utf-8"))
    assert lock["environment_revision"] == 9
    assert set(lock["required_sonames"]) == set(REQUIRED_SONAMES)


def test_old_v5_and_v7_certifications_are_rejected(tmp_path):
    marker = tmp_path / "old.json"
    for revision in (5, 7):
        marker.write_text(json.dumps({"status": "login_node_installation_certified",
            "a100_runtime_certified": False, "environment_revision": revision,
            "environment_prefix": str(tmp_path)}) + "\n", encoding="utf-8")
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / "require_certification.py"),
            "--kind", "login", "--path", str(marker), "--expected-prefix", str(tmp_path)],
            text=True, capture_output=True, check=False)
        assert result.returncode != 0 and "not passed" in result.stderr


def test_no_profile_approval_is_bundled():
    assert not list(ROOT.rglob("PROFILE_APPROVAL.json"))


def test_no_operational_script_uses_old_environment_prefix():
    paths = [ROOT / "setup_environment.sh", ROOT / "DELFTBLUE_RUNBOOK.md", *sorted((ROOT / "slurm").glob("*.sbatch"))]
    assert all("cu118_v5" not in path.read_text(encoding="utf-8") and "cu118_v7" not in path.read_text(encoding="utf-8") for path in paths)
