import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_primary_protocol_and_two_models_four_readouts():
    config = json.loads((ROOT / "PRIMARY_EXECUTION_CONFIG.json").read_text(encoding="utf-8"))
    assert config["official_outer_folds"] == list(range(5))
    assert config["seeds"] == list(range(5))
    assert len(config["trained_branches_per_cell"]) == 2
    assert len(config["readouts"]) == 4
    assert config["primary_comparison"] == "D_minus_B"
    assert config["inner_split"].startswith("frozen deterministic stratified 90/10")
    assert config["random2"] == {"synthetic_samples": 3000, "warmup_steps": 938, "batch_size": 128, "learning_rate": 0.0001}


def test_all_required_stages_and_resource_contracts_are_present():
    names = ["00_a100_preflight.sbatch", "05_build_graph_cache.sbatch", "06_smoke_fold0_seed0.sbatch",
        "07_profile_100_batches.sbatch", "08_primary_fold0_seed0.sbatch", "10_train_fold_seed.sbatch", "20_calibrate_export.sbatch",
        "30_consolidate_oof.sbatch", "40_final_audit.sbatch"]
    for name in names:
        source = (ROOT / "slurm" / name).read_text(encoding="utf-8")
        assert "--account=research-ME-mse" in source
        assert "--ntasks=1" in source
        assert "--cpus-per-task=8" in source
        assert "--mem-per-cpu=" in source
    for name in names[1:7]:
        source = (ROOT / "slurm" / name).read_text(encoding="utf-8")
        assert "--gpus-per-task=1" in source
        assert "--gres" not in source
        assert "--mem-per-cpu=8000M" in source


def test_setup_isolated_from_ambient_pip_configuration_before_conda():
    setup = (ROOT / "setup_environment.sh").read_text(encoding="utf-8")
    prefix = setup.split("command -v conda", 1)[0]
    assert "export PIP_CONFIG_FILE=/dev/null" in prefix
    for name in ("PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS", "PIP_NO_INDEX",
                 "PIP_TRUSTED_HOST", "PIP_CONSTRAINT", "PIP_REQUIREMENT"):
        assert name in prefix
    assert "PIP_DISABLE_PIP_VERSION_CHECK=1" in prefix


def test_gate_markers_and_final_counts_are_fail_closed():
    assert 'ALIGNN_ENVIRONMENT: PASS' in (ROOT / "slurm/00_a100_preflight.sbatch").read_text(encoding="utf-8")
    assert 'ALIGNN_GRAPH_CACHE: PASS' in (ROOT / "scripts/build_graph_cache.py").read_text(encoding="utf-8")
    assert 'ALIGNN_100_BATCH_PROFILE: PASS' in (ROOT / "scripts/profile_100_batches.py").read_text(encoding="utf-8")
    assert 'ALIGNN_FOLD0_SEED0_PRIMARY: PASS' in (ROOT / "slurm/08_primary_fold0_seed0.sbatch").read_text(encoding="utf-8")
    final = (ROOT / "scripts/final_audit.py").read_text(encoding="utf-8")
    for token in ("valid_cells", "trained_branches", "readouts", "25", "50", "100", "ALIGNN_PERSISTENT_FINAL_AUDIT: PASS"):
        assert token in final


def test_calibration_stage_verifies_then_atomically_promotes():
    source = (ROOT / "slurm/20_calibrate_export.sbatch").read_text(encoding="utf-8")
    assert source.index("verify_cell.py") < source.index("promote_cell.py")
    assert "--staging-root" in source and "--final-root" in source and "--quarantine-root" in source


def test_100_batch_and_primary_gates_block_scientific_grid():
    assert "require_profile_current.sh" in (ROOT / "slurm/08_primary_fold0_seed0.sbatch").read_text(encoding="utf-8")
    helper = (ROOT / "scripts/require_v37_external_gates.sh").read_text(encoding="utf-8")
    assert 'cd "$V26_ROOT"' in helper and "bash scripts/require_profile_current.sh" in helper
    assert "$V36_ROOT/scripts/require_primary_gate.py" in helper
    for name in ("10_train_fold_seed.sbatch", "20_calibrate_export.sbatch"):
        assert "require_v37_external_gates.sh" in (ROOT / "slurm" / name).read_text(encoding="utf-8")


def test_structure_cache_is_sharded_and_not_split_rebuilt():
    source = (ROOT / "scripts/build_graph_cache.py").read_text(encoding="utf-8")
    assert "SHARD_SIZE" in source and "STRUCTURE_CACHE_MANIFEST.json" not in source  # imported constant
    assert "for seed in" not in source and "for fold in" not in source
    assert '"graph_constructions_this_invocation"' in source
    assert '"split_specific_graph_copies": False' in source
    training = (ROOT / "alignn_stage2/training.py").read_text(encoding="utf-8")
    assert "cache_expected_provenance(dataset)" in training


def test_all_25_split_hashes_are_frozen_and_seed0_diagonal_is_reconciled():
    frozen = json.loads((ROOT / "ALL_25_SPLIT_HASHES.json").read_text(encoding="utf-8"))
    assert frozen["cell_count"] == 25
    assert {(row["fold"], row["seed"]) for row in frozen["cells"]} == {(f, s) for f in range(5) for s in range(5)}
    assert all(row["pairwise_disjoint"] and row["complete_outer_train_coverage"] for row in frozen["cells"])


def test_runbook_verification_and_failed_dependency_recovery_are_exact():
    runbook = (ROOT / "DELFTBLUE_RUNBOOK.md").read_text(encoding="utf-8")
    assert 'PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \\' in runbook
    assert "PYTHONDONTWRITEBYTECODE=1 \\" in runbook
    assert "python3 scripts/verify_package.py" in runbook
    for token in ("afterok", "scancel \"$CAL_JOB\"", "RETRY_JOB", "CAL_RETRY_JOB", "audit_grid_complete.py"):
        assert token in runbook


def test_outer_structure_policy_and_normalized_archive_permissions_are_explicit():
    policy = (ROOT / "OUTER_TEST_STRUCTURE_PREPROCESSING_POLICY.md").read_text(encoding="utf-8")
    assert "label-free graph preprocessing" in policy and "Outer-test labels" in policy
    permissions = json.loads((ROOT / "ARCHIVE_PERMISSION_POLICY.json").read_text(encoding="utf-8"))
    assert permissions["directories"] == "0755"
    assert permissions["executable_scripts"] == "0755"
    assert permissions["regular_files"] == "0644"
    assert permissions["world_writable_prohibited"] is True
    assert permissions["verified_from_tar_headers"] is True
