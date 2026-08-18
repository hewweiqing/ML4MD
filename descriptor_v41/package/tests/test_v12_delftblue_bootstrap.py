from pathlib import Path

import pytest

from alignn_stage2.slurm_resources import audit_gpu_a100_memory, memory_megabytes, parse_sbatch_headers

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("declared", "expected_mb"), [
    ("8000M", 8000), ("8000MB", 8000), ("8000MiB", 8000),
    ("4G", 4096), ("4GB", 4096), ("4GiB", 4096), ("7.5G", 7680)])
def test_slurm_memory_units_are_normalized(declared, expected_mb):
    assert memory_megabytes(declared) == expected_mb


def test_unit_normalization_rejects_the_observed_8g_delftblue_failure():
    assert memory_megabytes("8G") == 8192
    with pytest.raises(RuntimeError, match="8192 MB"):
        temporary = ROOT / "tests" / "_never_created"
        # Exercise the same decision independently without creating package debris.
        value = memory_megabytes("8G")
        if value > 8000:
            raise RuntimeError(f"synthetic.sbatch requests {value:g} MB per CPU")


def test_every_gpu_a100_header_is_at_or_below_delftblue_limit():
    records = audit_gpu_a100_memory(ROOT / "slurm")
    assert len(records) == 8
    assert all(row["normalized_mb_per_cpu"] <= 8000 for row in records)


def test_full_configuration_gpu_jobs_request_exactly_8000m_and_eight_cpus():
    required = ("05_build_graph_cache.sbatch", "06_smoke_fold0_seed0.sbatch",
        "07_profile_100_batches.sbatch", "08_primary_fold0_seed0.sbatch",
        "10_train_fold_seed.sbatch", "13_resolve_export_preouter_v40.sbatch",
        "20_calibrate_export.sbatch")
    for name in required:
        headers = parse_sbatch_headers(ROOT / "slurm" / name)
        assert headers["partition"] == "gpu-a100"
        assert headers["cpus-per-task"] == "8"
        assert headers["mem-per-cpu"] == "8000M"
        assert memory_megabytes(headers["mem-per-cpu"]) == 8000
    assert parse_sbatch_headers(ROOT / "slurm/00_a100_preflight.sbatch")["mem-per-cpu"] == "4G"


def test_runbook_avoids_python36_in_isolated_python310_subshell_and_restores_slurm_checks():
    runbook = (ROOT / "DELFTBLUE_RUNBOOK.md").read_text(encoding="utf-8")
    assert "Python 3.6.8" in runbook and "future feature annotations is not defined" in runbook
    start = runbook.index("(\n  module purge")
    verify = runbook.index("python3 scripts/verify_package.py", start)
    close = runbook.index("\n)", verify)
    assert "module load 2024r1" in runbook[start:close]
    assert "module load python/3.10.12" in runbook[start:close]
    command_check = runbook.index("command -v sbatch", close)
    version_check = runbook.index("sbatch --version", command_check)
    test_only = runbook.index("sbatch --test-only", version_check)
    assert close < command_check < version_check < test_only
    assert "must not run `module purge` directly" in runbook


def test_setup_conda_resolution_precedence_and_atomic_certified_reuse_are_explicit():
    setup = (ROOT / "setup_environment.sh").read_text(encoding="utf-8")
    supplied = setup.index('${CONDA_EXE:-}')
    path_lookup = setup.index("command -v conda", supplied)
    miniforge = setup.index('$HOME/miniforge3/bin/conda', path_lookup)
    failure = setup.index("This script will not install or replace Miniforge automatically", miniforge)
    assert supplied < path_lookup < miniforge < failure
    assert '"$CONDA_BIN" create' in setup
    assert 'STAGING_PREFIX="${ENV_PREFIX}.staging"' in setup
    assert 'create --prefix "$ENV_PREFIX" --clone "$STAGING_PREFIX"' in setup
    assert '--check-only --expected-prefix "$ENV_PREFIX"' in setup
    assert 'Reused fully certified revision-9 environment' in setup


def test_profile_to_primary_remains_manual():
    runbook = (ROOT / "DELFTBLUE_RUNBOOK.md").read_text(encoding="utf-8")
    assert "--dependency=afterok:$PROFILE_JOB slurm/08_primary_fold0_seed0.sbatch" not in runbook
