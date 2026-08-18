from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V26_AGGREGATE = "1c15f877cee4eb6eebfd096962c6a793d0d3f14385287a6a9385e3adce05bd7d"
OLD_MUBEN = "868654151b87d526f122f741183c23599913541d5d62aada5cb35c91bc510719"


def test_revised_scaler_is_hash_pinned_and_not_the_old_source():
    approval = json.loads((ROOT / "MUBEN_TS_APPROVAL.json").read_text(encoding="utf-8"))
    source = ROOT / approval["installed_source"]
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    assert actual == approval["expected_sha256"]
    assert actual != OLD_MUBEN
    assert approval["supersedes_source_sha256"] == OLD_MUBEN
    assert approval["selection_used_outer_test"] is False
    assert approval["outer_test_access_started_before_amendment"] is False


def test_convergence_rule_uses_one_explicit_final_gradient_tolerance():
    source = (ROOT / "vendor/muben_temperature_scaling.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"TOLERANCE_GRAD", "MAX_ITERATIONS"}}
    assert assignments == {"MAX_ITERATIONS": 500, "TOLERANCE_GRAD": 1e-7}
    assert "gradient <= TOLERANCE_GRAD\n" in source
    assert "TOLERANCE_GRAD * 10" not in source
    assert "T=exp(log_T)" in source
    assert "dtype=torch.float64" in source
    assert "shape [n, 2]" in source


def test_v26_cache_identity_is_explicitly_pinned_for_artifact_reuse():
    source = (ROOT / "alignn_stage2/training.py").read_text(encoding="utf-8")
    assert f'V26_CACHE_RELEASE_IDENTITY = "{V26_AGGREGATE}"' in source
    assert "release_identity=V26_CACHE_RELEASE_IDENTITY" in source


def test_recovery_checks_validation_before_outer_export_and_never_trains():
    slurm = (ROOT / "slurm/09_recover_calibration_f0s0.sbatch").read_text(encoding="utf-8")
    assert "validate_v31_calibration_recovery.py" in slurm
    assert "validate_v35_existing_export.py" in slurm
    assert slurm.index("validate_v31_calibration_recovery.py") < slurm.index("calibrate_and_export.py")
    assert "train_paired.py" not in slurm
    assert "--partition=gpu-a100-small" in slurm
    assert "--time=01:00:00" in slurm
    assert "--cpus-per-task=2" in slurm


def test_validation_helper_refuses_existing_outer_sentinel():
    source = (ROOT / "scripts/validate_v31_calibration_recovery.py").read_text(encoding="utf-8")
    assert 'cell / "OUTER_TEST_ACCESS_STARTED.json"' in source
    assert "automatic recovery is prohibited" in source
    assert '"outer_test_accessed": False' in source


def test_v33_ordering_check_is_tolerance_aware_and_still_fail_closed():
    source = (ROOT / "alignn_stage2/calibration_contract.py").read_text(encoding="utf-8")
    assert "def verify_binary_ordering_with_tolerance" in source
    assert "np.diff(ordered)" in source
    assert "BINARY_EQUIVALENCE_ATOL + BINARY_EQUIVALENCE_RTOL" in source
    assert "differences < -allowance" in source
    assert "ordering changed beyond declared numerical tolerance" in source
    assert "np.array_equal(raw_order, scaled_order)" not in source


def test_v40_gpu_partition_counts_include_three_short_recovery_jobs():
    source = (ROOT / "tests/test_v12_delftblue_bootstrap.py").read_text(encoding="utf-8")
    assert "assert len(records) == 8" in source
    regular_gpu_jobs = [path for path in (ROOT / "slurm").glob("*.sbatch")
        if "#SBATCH --partition=gpu-a100\n" in path.read_text(encoding="utf-8")]
    small_gpu_jobs = [path for path in (ROOT / "slurm").glob("*.sbatch")
        if "#SBATCH --partition=gpu-a100-small\n" in path.read_text(encoding="utf-8")]
    assert len(regular_gpu_jobs) == 8
    assert len(small_gpu_jobs) == 3


def test_v35_existing_export_path_never_rematerializes_outer_test():
    slurm = (ROOT / "slurm/09_recover_calibration_f0s0.sbatch").read_text(encoding="utf-8")
    assert 'if [[ -f "$CELL/OUTER_TEST_ACCESS_STARTED.json" ]]' in slurm
    existing_branch = slurm.split("else", 1)[0]
    assert "validate_v35_existing_export.py" in existing_branch
    assert "calibrate_and_export.py" not in existing_branch
    helper = (ROOT / "scripts/validate_v35_existing_export.py").read_text(encoding="utf-8")
    assert "outer_test_rematerialized\": False" in helper
    assert "outer_test_metrics_inspected_by_recovery_validator\": False" in helper


def test_production_verifier_uses_tolerance_ordering_without_quadratic_tie_matrix():
    source = (ROOT / "alignn_stage2/production.py").read_text(encoding="utf-8")
    assert "verify_binary_ordering_with_tolerance(raw, scaled)" in source
    assert "verify_binary_ordering_with_tolerance(scaled, raw)" in source
    assert "raw_margin[:, None]" not in source
    assert "scaled_margin[:, None]" not in source
