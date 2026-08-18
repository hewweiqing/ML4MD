from datetime import datetime, timezone

import pytest

from alignn_stage2 import stage_a_profile


def _valid_profile(**overrides) -> dict:
    profile = {"status": "passed", "fold": 0, "seed": 0, "elapsed_seconds": 120.0,
        "elapsed_seconds_per_replay": 60.0, "optimizer_steps_per_replay": 938, "batch_size": 128,
        "deterministic_replay": True, "projected_seconds_for_20_seeds": 2400.0,
        "package_aggregate_sha256": "a" * 64, "diagnostics": {"changed_state_key_count": 5, "step_count": 938}}
    profile.update(overrides)
    return profile


def test_validate_profile_accepts_a_well_formed_report():
    stage_a_profile.validate_profile(_valid_profile())


def test_validate_profile_rejects_non_deterministic_replay():
    with pytest.raises(RuntimeError, match="deterministic replay"):
        stage_a_profile.validate_profile(_valid_profile(deterministic_replay=False))


def test_validate_profile_rejects_wrong_optimizer_step_budget():
    with pytest.raises(RuntimeError, match="optimizer-step budget"):
        stage_a_profile.validate_profile(_valid_profile(optimizer_steps_per_replay=100))


def test_validate_profile_rejects_wrong_batch_size():
    with pytest.raises(RuntimeError, match="batch size"):
        stage_a_profile.validate_profile(_valid_profile(batch_size=32))


def test_evaluate_policy_passes_under_budget_and_fails_over_budget():
    policy = {"projected_wall_time_hours_max": 1.0}
    under_budget = stage_a_profile.evaluate_policy(_valid_profile(projected_seconds_for_20_seeds=1800), policy)
    assert under_budget["passed"] is True
    over_budget = stage_a_profile.evaluate_policy(_valid_profile(projected_seconds_for_20_seeds=7200), policy)
    assert over_budget["passed"] is False


def test_verify_approval_requires_matching_bindings_and_policy_evaluation():
    profile = _valid_profile()
    policy = {"projected_wall_time_hours_max": 24.0}
    bindings = {"profile_sha256": "b" * 64, "package_aggregate_sha256": profile["package_aggregate_sha256"]}
    evaluation = stage_a_profile.evaluate_policy(profile, policy)
    approval = {"schema_version": 1, "decision": "approved", "reviewer_identity": "someone",
        "reasons": ["under budget"], "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": bindings, "resource_policy_evaluation": evaluation}
    stage_a_profile.verify_approval(approval, profile, policy, bindings)


def test_verify_approval_rejects_stale_bindings():
    profile = _valid_profile()
    policy = {"projected_wall_time_hours_max": 24.0}
    bindings = {"profile_sha256": "b" * 64, "package_aggregate_sha256": profile["package_aggregate_sha256"]}
    evaluation = stage_a_profile.evaluate_policy(profile, policy)
    approval = {"schema_version": 1, "decision": "approved", "reviewer_identity": "someone",
        "reasons": ["under budget"], "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": bindings, "resource_policy_evaluation": evaluation}
    stale_bindings = {"profile_sha256": "c" * 64, "package_aggregate_sha256": profile["package_aggregate_sha256"]}
    with pytest.raises(RuntimeError, match="stale"):
        stage_a_profile.verify_approval(approval, profile, policy, stale_bindings)


def test_verify_approval_rejects_missing_reviewer_identity():
    profile = _valid_profile()
    policy = {"projected_wall_time_hours_max": 24.0}
    bindings = {"profile_sha256": "b" * 64, "package_aggregate_sha256": profile["package_aggregate_sha256"]}
    evaluation = stage_a_profile.evaluate_policy(profile, policy)
    approval = {"schema_version": 1, "decision": "approved", "reviewer_identity": "",
        "reasons": ["under budget"], "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": bindings, "resource_policy_evaluation": evaluation}
    with pytest.raises(RuntimeError, match="reviewer identity"):
        stage_a_profile.verify_approval(approval, profile, policy, bindings)
