from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def text(name):
    return (ROOT / "slurm" / name).read_text(encoding="utf-8")


def test_remaining_arrays_exclude_completed_cell_zero():
    for name in ("10_train_fold_seed.sbatch", "20_calibrate_export.sbatch"):
        source = text(name)
        assert "#SBATCH --array=1-24%5" in source
        assert "#SBATCH --array=0-24" not in source


def test_external_approved_gates_are_reused_without_rerunning_them():
    helper = (ROOT / "scripts/require_v37_external_gates.sh").read_text(encoding="utf-8")
    assert "$V26_ROOT/preflight/A100_RUNTIME_CERTIFICATION.json" in helper
    assert 'cd "$V26_ROOT"' in helper and "bash scripts/require_profile_current.sh" in helper
    assert "$V36_ROOT/preflight/PRIMARY_FOLD0_GATE_PASSED.json" in helper
    assert "validate_v37_full_grid_inputs.py" in helper


def test_training_and_calibration_remain_separate():
    train = text("10_train_fold_seed.sbatch")
    calibrate = text("20_calibrate_export.sbatch")
    assert "train_paired.py" in train and "calibrate_and_export.py" not in train
    assert calibrate.index("calibrate_and_export.py") < calibrate.index("verify_cell.py") < calibrate.index("promote_cell.py")
