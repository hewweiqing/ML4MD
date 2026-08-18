from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from alignn_stage2.sigma_authorization import (
    CoordinateSigmaAuthorizationMissing,
    require_authorized_sigma,
)
from alignn_stage2.slurm_resources import audit_slurm_resources

ROOT = Path(__file__).resolve().parents[1]


def test_all_python_and_json_files_are_parseable():
    for path in ROOT.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for path in ROOT.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))


def test_gpu_slurm_workflow_and_resource_policy():
    rows = audit_slurm_resources(ROOT / "slurm")
    assert len(rows) == 8
    assert sum(row["partition"].startswith("gpu-a100") for row in rows) == 6
    assert sum(row["partition"] == "compute" for row in rows) == 2
    assert all(row["gpus_per_task"] == 1 for row in rows if row["partition"].startswith("gpu-a100"))


def test_gpu_training_is_cuda_native_and_cpu_checkpoint_roots_are_rejected():
    source = (ROOT / "alignn_stage2/gpu_coordinate_training.py").read_text(encoding="utf-8")
    wrapper = (ROOT / "scripts/run_gpu_coordinate_cell.sh").read_text(encoding="utf-8")
    assert "configure_gpu_runtime()" in source
    assert ".to(DEVICE)" in source
    assert "torch_cuda_rng_state_all" in source
    assert "paired_gpu_base.pt" in source
    assert "coordinate_gpu_warmup_only.pt" in source
    assert "cpu_coordinate*|*stage2_primary_v26*" in wrapper
    assert "gpu_coordinate_primary_v4" in wrapper


def test_job_10654092_cpu_permutation_order_is_preserved_before_cuda_indexing():
    source = (ROOT / "alignn_stage2/gpu_coordinate_training.py").read_text(encoding="utf-8")
    assert 'torch.Generator(device="cpu").manual_seed(seed + 280000)' in source
    assert "torch.Generator(device=DEVICE).manual_seed(seed + 280000)" not in source
    assert source.count("torch.randperm(3000, generator=generator)") == 2
    assert "cpu_indices, cursor = permutation[cursor:cursor + 128], cursor + 128" in source
    assert "indices = cpu_indices.to(device=DEVICE, non_blocking=False)" in source
    assert source.index("cpu_indices, cursor =") < source.index("indices = cpu_indices.to(")


def test_scientific_design_and_fixed_dose_are_preserved():
    config = json.loads((ROOT / "GPU_COORDINATE_FULL_CONFIG.json").read_text(encoding="utf-8"))
    amendment = (ROOT / "GPU_COORDINATE_PROSPECTIVE_DEVICE_AMENDMENT.md").read_text(encoding="utf-8")
    assert config["folds"] == [0, 1, 2, 3, 4]
    assert config["seeds"] == [0, 1, 2, 3, 4]
    assert config["models_per_cell"] == 2
    assert len(config["conditions"]) == 4
    assert "0.040" in amendment and "project_defined_fixed_dose" in amendment
    assert "CPU checkpoints prohibited" in json.dumps(config)


def test_approved_muben_scaler_is_preserved_exactly():
    digest = hashlib.sha256((ROOT / "vendor/muben_temperature_scaling.py").read_bytes()).hexdigest()
    assert digest == "108b3183400725ca53e025c6f0f92690f35069f6fd786091b9d37dc50e90f08b"


def test_outer_test_is_absent_from_training_and_deferred_to_export():
    training = (ROOT / "alignn_stage2/gpu_coordinate_training.py").read_text(encoding="utf-8")
    export = (ROOT / "alignn_stage2/gpu_coordinate_calibrate_export.py").read_text(encoding="utf-8")
    assert "load_outer_test" not in training
    assert "OUTER_TEST_ACCESS_STARTED.json" in export
    assert export.index("fit_temperature(") < export.index("OUTER_TEST_ACCESS_STARTED.json")


def test_arrays_cover_primary_plus_remaining_24_cells():
    primary = (ROOT / "slurm/08_primary_coordinate_f0s0_gpu.sbatch").read_text(encoding="utf-8")
    grid = (ROOT / "slurm/10_train_coordinate_grid_gpu.sbatch").read_text(encoding="utf-8")
    calibration = (ROOT / "slurm/20_calibrate_export_coordinate_gpu.sbatch").read_text(encoding="utf-8")
    assert "run_gpu_coordinate_cell.sh 0 0" in primary
    assert "#SBATCH --array=1-24%5" in grid
    assert "#SBATCH --array=0-24%5" in calibration


def test_exact_external_v29_authorization_is_bound_without_rewriting(tmp_path):
    fixture = ROOT / "tests_gpu/fixtures/SELECTED_COORDINATE_SIGMA_V29.json"
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == (
        "ea817b08ec5e5d150307e4a26fcbeb561b318c666a33612ee9b2b0182e624fc8"
    )
    authorized = require_authorized_sigma(authorization_path=fixture)
    assert authorized.sigma == 0.04
    assert authorized.package_aggregate_sha256 == (
        "d91ff9b17b6a75d8b38ecb62efbdbea796560426c365f8bb3a704091606183ef"
    )

    altered = tmp_path / "altered.json"
    value = json.loads(fixture.read_text(encoding="utf-8"))
    value["sigma_cartesian_per_axis_angstrom"] = 0.041
    altered.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(CoordinateSigmaAuthorizationMissing):
        require_authorized_sigma(authorization_path=altered)


def test_authorization_binding_is_scientifically_outcome_neutral():
    binding = json.loads((ROOT / "GPU_COORDINATE_V29_AUTHORIZATION_BINDING.json").read_text())
    assert binding["authorization_origin_release"] == "delftblue_package_v29"
    assert binding["selection_used_alignn_validation_metrics"] is False
    assert binding["selection_used_alignn_outer_test"] is False
    assert binding["selection_used_calibration_results"] is False
    assert binding["outer_test_results_accessed_for_binding"] is False


def test_batch_cuda_activation_helper_is_present_and_certified_lineage_exact():
    helper = ROOT / "scripts/verify_cuda_runtime.py"
    assert helper.is_file()
    assert hashlib.sha256(helper.read_bytes()).hexdigest() == (
        "656c167f228562f3e1e70069ee15853e8025f301586402b3c583191e21b968f6"
    )
    activation = (ROOT / "scripts/activate_cuda_runtime.sh").read_text(encoding="utf-8")
    assert '"$_runtime_helper_dir/verify_cuda_runtime.py"' in activation
