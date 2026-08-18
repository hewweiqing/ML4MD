from pathlib import Path
from types import SimpleNamespace

import pytest

from alignn_stage2.slurm_live_validation import run_test_only_validation
from alignn_stage2.slurm_resources import audit_slurm_resources, parse_sbatch_headers

ROOT = Path(__file__).resolve().parents[1]


def write_sbatch(path, *, partition="compute", memory="3968M", ntasks="1", cpus="8",
        nodes=False, exclusive=False):
    lines = ["#!/bin/bash", f"#SBATCH --partition={partition}"]
    if nodes:
        lines.append("#SBATCH --nodes=1")
    if exclusive:
        lines.append("#SBATCH --exclusive")
    if ntasks is not None:
        lines.append(f"#SBATCH --ntasks={ntasks}")
    if cpus is not None:
        lines.append(f"#SBATCH --cpus-per-task={cpus}")
    lines.append(f"#SBATCH --mem-per-cpu={memory}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_all_thirteen_jobs_are_audited_with_partition_specific_limits_and_no_nodes():
    records = audit_slurm_resources(ROOT / "slurm")
    assert len(records) == 13
    compute = [row for row in records if row["partition"] == "compute"]
    assert {row["file"] for row in compute} == {"30_consolidate_oof.sbatch", "40_final_audit.sbatch"}
    assert all(row["declared_memory_per_cpu"] == "3968M" for row in compute)
    assert all(row["normalized_mb_per_cpu"] == 3968 for row in compute)
    assert all(row["maximum_mb_per_cpu"] == 3968 for row in compute)
    assert all(row["ntasks"] == 1 for row in records)
    small_jobs = {"09_recover_calibration_f0s0.sbatch", "11_diagnose_preouter_convergence_v38.sbatch",
        "12_recover_existing_exports_v38.sbatch"}
    ordinary = [row for row in records if row["file"] not in small_jobs]
    assert all(row["cpus_per_task"] == 8 for row in ordinary)
    assert all(row["nodes_declared"] is False for row in records)
    recovery = [row for row in records if row["file"] in small_jobs]
    assert len(recovery) == 3
    assert all(row["partition"] == "gpu-a100-small" for row in recovery)
    assert all(row["cpus_per_task"] == 2 for row in recovery)
    assert all(row["maximum_mb_per_cpu"] == 8000 for row in recovery)


@pytest.mark.parametrize(("partition", "memory", "message"), [
    ("compute", "4G", "4096 MB per CPU"),
    ("compute-p1", "4GiB", "4096 MB per CPU"),
    ("gpu-a100", "8GB", "8192 MB per CPU")])
def test_partition_memory_overages_are_rejected_behaviorally(tmp_path, partition, memory, message):
    write_sbatch(tmp_path / "bad.sbatch", partition=partition, memory=memory)
    with pytest.raises(RuntimeError, match=message):
        audit_slurm_resources(tmp_path)


def test_nonexclusive_single_task_nodes_directive_is_rejected(tmp_path):
    write_sbatch(tmp_path / "bad.sbatch", nodes=True)
    with pytest.raises(RuntimeError, match="unnecessary --nodes"):
        audit_slurm_resources(tmp_path)
    (tmp_path / "bad.sbatch").unlink()
    write_sbatch(tmp_path / "exclusive.sbatch", nodes=True, exclusive=True)
    assert audit_slurm_resources(tmp_path)[0]["exclusive"] is True


@pytest.mark.parametrize("missing", ["ntasks", "cpus"])
def test_ntasks_and_cpus_per_task_are_required(tmp_path, missing):
    write_sbatch(tmp_path / "bad.sbatch", ntasks=None if missing == "ntasks" else "1",
        cpus=None if missing == "cpus" else "8")
    with pytest.raises(RuntimeError, match="requires a positive"):
        audit_slurm_resources(tmp_path)


class FakeRunner:
    def __init__(self, *, returncode=0, stdout="sbatch: Job 12345 to start at 2026-01-01\n", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append((command, kwargs))
        return SimpleNamespace(returncode=self.returncode, stdout=self.stdout, stderr=self.stderr)


def test_live_helper_runs_test_only_for_every_script_and_treats_ids_as_simulations():
    runner = FakeRunner()
    records = run_test_only_validation(ROOT / "slurm", runner=runner, sbatch_executable="/usr/bin/sbatch")
    assert len(records) == 13 and all(row["simulation_only"] for row in records)
    assert all(command[0][1] == "--test-only" for command in runner.commands)
    assert all(command[1] == {"text": True, "capture_output": True, "check": False} for command in runner.commands)


@pytest.mark.parametrize(("returncode", "output", "token"), [
    (1, "", "returncode=1"), (0, "error: invalid account", "error:"),
    (0, "allocation failure", "allocation failure"),
    (0, "requests resources on a per-node basis and does not request complete nodes", "per-node basis")])
def test_live_helper_fails_on_status_errors_allocation_or_per_node_warning(returncode, output, token):
    runner = FakeRunner(returncode=returncode, stderr=output)
    with pytest.raises(RuntimeError, match=token):
        run_test_only_validation(ROOT / "slurm", runner=runner, sbatch_executable="sbatch")


def test_operator_helper_and_runbook_are_explicitly_simulation_only_and_import_safe():
    helper = (ROOT / "scripts/delftblue_test_only.py").read_text(encoding="utf-8")
    assert "SIMULATION ONLY" in helper and "submits no jobs" in helper and '"jobs_submitted": 0' in helper
    runbook = (ROOT / "DELFTBLUE_RUNBOOK.md").read_text(encoding="utf-8")
    corrected = ('PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \\\n'
        'PYTHONDONTWRITEBYTECODE=1 \\\n'
        '"$ALIGNN_ENV_PREFIX/bin/python" scripts/audit_slurm_resources.py')
    assert corrected in runbook
    assert "scripts/delftblue_test_only.py" in runbook
    assert "for f in slurm/*.sbatch; do sbatch --test-only" not in runbook


def test_corrected_headers_preserve_required_nonmemory_directives():
    for path in sorted((ROOT / "slurm").glob("*.sbatch")):
        headers = parse_sbatch_headers(path)
        assert headers["account"] == "research-ME-mse"
        assert headers["ntasks"] == "1"
        expected_cpus = "2" if path.name in {"09_recover_calibration_f0s0.sbatch",
            "11_diagnose_preouter_convergence_v38.sbatch", "12_recover_existing_exports_v38.sbatch"} else "8"
        assert headers["cpus-per-task"] == expected_cpus
        assert "time" in headers and "nodes" not in headers
