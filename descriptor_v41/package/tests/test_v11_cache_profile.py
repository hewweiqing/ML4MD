import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from alignn_stage2.common import sha256_file
from alignn_stage2.resource_profile import binding_values, evaluate_policy, validate_profile, verify_approval
from alignn_stage2.structure_cache import (MANIFEST_NAME, atomic_write_shard, canonical_digest,
    expected_provenance, load_selected, rebuild_invalid_shard, validate_manifest, verify_cache)


class FakeDGL:
    __version__ = "fake-dgl-1"

    @staticmethod
    def save_graphs(path, graphs):
        Path(path).write_text(json.dumps(list(graphs)), encoding="utf-8")

    @staticmethod
    def load_graphs(path):
        return json.loads(Path(path).read_text(encoding="utf-8")), {}


def cache_fixture(tmp_path):
    root, quarantine = tmp_path / "cache", tmp_path / "quarantine"
    root.mkdir(); quarantine.mkdir()
    provenance = expected_provenance({"cutoff": 8.0}, jarvis_version="fake-jarvis-1",
        dgl_version=FakeDGL.__version__, release_identity="a" * 64)
    ids, hashes = ["s1", "s2"], ["1" * 64, "2" * 64]
    record = atomic_write_shard(root, 0, ["a1", "a2"], ["l1", "l2"], ids, hashes,
        provenance_sha256=provenance["provenance_sha256"], dgl_module=FakeDGL)
    entries = {structure_id: {"shard": 0, "offset": offset, "structure_sha256": hashes[offset]}
        for offset, structure_id in enumerate(ids)}
    manifest = {"schema_version": 2, "status": "passed", "provenance": provenance,
        "release_identity": provenance["release_identity"], "dataset_structure_count": 2,
        "structure_count": 2, "structure_ids_sha256": canonical_digest(sorted(entries)),
        "entries": entries, "shards": [record],
        "exhaustive_verification": {"status": "passed", "verified_shards": 1, "verified_graphs": 2}}
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return root, quarantine, provenance, record, manifest


def test_cache_manifest_has_complete_provenance_and_exhaustive_verification(tmp_path):
    root, _, provenance, _, manifest = cache_fixture(tmp_path)
    validate_manifest(manifest, provenance)
    result = verify_cache(root, provenance, dgl_module=FakeDGL)
    assert result["status"] == "passed" and result["verified_graphs"] == 2
    for key in ("dataset_sha256", "graph_settings", "graph_settings_sha256", "alignn_commit",
            "jarvis_tools_version", "dgl_version", "shard_size", "scope", "label_free_policy",
            "release_identity", "provenance_sha256"):
        assert key in manifest["provenance"]


@pytest.mark.parametrize("changed", ["dataset_sha256", "graph_settings", "jarvis_tools_version", "dgl_version", "release_identity"])
def test_load_selected_rejects_wrong_provenance(tmp_path, monkeypatch, changed):
    root, _, provenance, _, _ = cache_fixture(tmp_path)
    monkeypatch.setitem(sys.modules, "dgl", FakeDGL)
    wrong = copy.deepcopy(provenance)
    wrong[changed] = {"wrong": True} if changed == "graph_settings" else "wrong"
    wrong["provenance_sha256"] = canonical_digest({k: v for k, v in wrong.items() if k != "provenance_sha256"})
    with pytest.raises(RuntimeError, match="provenance"):
        load_selected(root, ["s1"], wrong)


@pytest.mark.parametrize("corrupt_name", ["atom_00000.bin", "line_00000.bin", "shard_00000.json"])
def test_corrupt_shard_component_is_quarantined_and_atomically_rebuilt(tmp_path, corrupt_name):
    root, quarantine, provenance, _, _ = cache_fixture(tmp_path)
    (root / corrupt_name).write_text("corrupt", encoding="utf-8")
    record, destination = rebuild_invalid_shard(root, quarantine, 0, atom_graphs=["new-a1", "new-a2"],
        line_graphs=["new-l1", "new-l2"], ids=["s1", "s2"], structure_hashes=["1" * 64, "2" * 64],
        provenance_sha256=provenance["provenance_sha256"], dgl_module=FakeDGL)
    assert destination is not None and (destination / corrupt_name).is_file()
    assert FakeDGL.load_graphs(root / record["atom_file"])[0][0] == "new-a1"
    assert not list(root.glob("*.tmp.*"))


def test_interrupted_partial_shard_is_quarantined_and_rebuilt(tmp_path):
    root, quarantine = tmp_path / "cache", tmp_path / "quarantine"
    root.mkdir(); quarantine.mkdir()
    (root / "atom_00000.bin").write_text("partial evidence", encoding="utf-8")
    provenance_sha = "a" * 64
    record, destination = rebuild_invalid_shard(root, quarantine, 0, atom_graphs=["a"], line_graphs=["l"],
        ids=["s1"], structure_hashes=["1" * 64], provenance_sha256=provenance_sha, dgl_module=FakeDGL)
    assert destination is not None and (destination / "atom_00000.bin").read_text() == "partial evidence"
    assert record["graph_count"] == 1 and (root / "shard_00000.json").is_file()


def valid_profile():
    return {"schema_version": 2, "status": "passed", "package_aggregate_sha256": "a" * 64,
        "dataset_sha256": "b" * 64, "split_identity_sha256": "c" * 64,
        "cache_manifest_sha256": "d" * 64, "cache_provenance_sha256": "e" * 64,
        "a100_runtime_certification_sha256": "f" * 64, "execution_plan_sha256": "1" * 64,
        "resource_policy_sha256": "2" * 64, "primary_slurm_sha256": "3" * 64,
        "cache": {"total_verified_load_seconds": 10.0, "requested_structures": 1000,
            "manifest_lookup_seconds": .1, "hash_verification_seconds": 1.0, "graph_loading_seconds": 8.0,
            "requested_shards": 2, "verified_shards": 2, "memory_cache_hits": 0, "memory_cache_misses": 2},
        "initialization": {"model_optimizer_scheduler_seconds": 2.0},
        "training": {"measured_batches": 100, "finite_losses": True,
            "total_measured_seconds": 100.0, "seconds_per_batch": 1.0,
            "forward_seconds": 30.0, "backward_seconds": 40.0, "optimizer_seconds": 20.0, "scheduler_seconds": 1.0},
        "random2": {"descriptor_measured_seconds": 5.0, "descriptor_projected_seconds": 50.0,
            "deterministic_replay_verified": True,
            "warmup_replay_measured_seconds": 20.0},
        "validation": {"representative_full_validation_seconds": 30.0,
            "scheduled_evaluations_all_branches": 84},
        "io": {"checkpoint_history_measured_seconds": 1.0},
        "memory": {"peak_cuda_allocated_bytes": 10, "peak_cuda_reserved_bytes": 20,
            "detected_gpu_total_bytes": 100, "process_peak_rss_bytes": 10,
            "projected_host_peak_bytes": 20},
        "disk": {"cache_size_bytes": 100, "projected_cell_output_bytes": 100,
            "projected_full_grid_bytes": 2600, "available_bytes": 10000},
        "projections": {"control_branch_hours": 1.0, "random2_branch_hours": 1.5,
            "paired_cell_hours": 3.0, "time_safety_factor": 1.25,
            "conservative_paired_upper_bound_hours": 3.75},
        "measured_phases": ["graph_cache_verified_load", "100_optimizer_batches"],
        "excluded_phases": ["structure_graph_construction", "outer-test access"],
        "calculation_formulas": {"paired": "cache + control + random2"}}


def policy_fixture():
    return {"approval_thresholds": {"runtime_fraction_max": .8, "host_memory_fraction_max": .8,
        "gpu_reserved_fraction_max": .8, "disk_fraction_max": .8},
        "requested_slurm_resources": {"primary_wall_time_hours": 24, "host_memory_bytes": 100},
        "profile": {"paired_time_safety_factor": 1.25}}


def test_profile_requires_truthful_phases_paired_overhead_and_resource_measurements():
    report = valid_profile(); validate_profile(report)
    false_claim = copy.deepcopy(report); false_claim["measured_phases"].append("structure_graph_construction")
    with pytest.raises(RuntimeError, match="falsely claims"): validate_profile(false_claim)
    omitted_branch = copy.deepcopy(report); omitted_branch["projections"]["paired_cell_hours"] = 2.0
    omitted_branch["projections"]["conservative_paired_upper_bound_hours"] = 2.5
    with pytest.raises(RuntimeError, match="both branches"): validate_profile(omitted_branch)
    missing_disk = copy.deepcopy(report); del missing_disk["disk"]["available_bytes"]
    with pytest.raises(RuntimeError, match="available_bytes"): validate_profile(missing_disk)


def test_profile_timer_starts_before_graph_loading_and_no_approval_is_bundled():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/profile_100_batches.py").read_text(encoding="utf-8")
    assert source.index("cache_phase_started = time.perf_counter()") < source.index(
        'build_or_load_graphs(dataset, train_ids')
    assert not (root / "PROFILE_APPROVAL.json").exists()
    assert not (root / "preflight/PROFILE_APPROVAL.json").exists()


def test_profile_output_path_cannot_be_shadowed_by_model_tensor():
    """Regression for DelftBlue job 10633397, which failed after all 100 batches."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/profile_100_batches.py").read_text(encoding="utf-8")
    assert "dataset, cache_root, output_path =" in source
    assert "model_output, elapsed = synchronized_timing(lambda: model(batch))" in source
    assert "nll_loss(model_output, labels)" in source
    assert "output_path.parent.mkdir(parents=True, exist_ok=True)" in source
    assert "os.replace(temporary, output_path)" in source


def approval_fixture(tmp_path):
    profile = valid_profile(); policy = policy_fixture()
    files = {"profile": tmp_path / "profile.json", "cert": tmp_path / "cert.json",
        "package": tmp_path / "package.json", "plan": tmp_path / "plan.json",
        "cache": tmp_path / "cache.json", "policy": tmp_path / "policy.json", "slurm": tmp_path / "primary.sbatch"}
    files["cert"].write_text('{"status":"passed","a100_runtime_certified":true}')
    files["package"].write_text(json.dumps({"aggregate_sha256": "a" * 64}))
    files["plan"].write_text("plan")
    files["cache"].write_text(json.dumps({"status": "passed", "exhaustive_verification": {"status": "passed"}}))
    files["slurm"].write_text("slurm")
    files["policy"].write_text(json.dumps(policy))
    profile.update({"a100_runtime_certification_sha256": sha256_file(files["cert"]),
        "execution_plan_sha256": sha256_file(files["plan"]), "cache_manifest_sha256": sha256_file(files["cache"]),
        "resource_policy_sha256": sha256_file(files["policy"]), "primary_slurm_sha256": sha256_file(files["slurm"]),
        "requested_slurm_resources": policy["requested_slurm_resources"]})
    files["profile"].write_text(json.dumps(profile))
    bindings = binding_values(files["profile"], files["cert"], files["package"], files["plan"],
        files["cache"], files["policy"], files["slurm"])
    evaluation = evaluate_policy(profile, policy)
    approval = {"schema_version": 1, "decision": "approved", "reviewer_identity": "test-reviewer",
        "approved_at_utc": "2026-07-20T12:00:00+00:00", "reasons": ["all margins pass"],
        "bindings": bindings, "resource_policy_evaluation": evaluation}
    return files, profile, policy, approval


def test_approval_is_bound_to_exact_profile_certification_package_cache_and_resources(tmp_path):
    files, profile, policy, approval = approval_fixture(tmp_path)
    verify_approval(approval, profile, policy, approval["bindings"])
    for name in ("profile", "cert", "package", "cache", "policy", "slurm"):
        changed = copy.deepcopy(approval["bindings"])
        changed_key = {"profile": "profile_sha256", "cert": "a100_runtime_certification_sha256",
            "package": "package_aggregate_sha256", "cache": "cache_manifest_sha256",
            "policy": "resource_policy_sha256", "slurm": "primary_slurm_sha256"}[name]
        changed[changed_key] = "9" * 64
        with pytest.raises(RuntimeError, match="stale, copied, or mismatched"):
            verify_approval(approval, profile, policy, changed)


def test_missing_rejected_or_failed_policy_approval_blocks_execution(tmp_path):
    _, profile, policy, approval = approval_fixture(tmp_path)
    rejected = copy.deepcopy(approval); rejected["decision"] = "rejected"
    with pytest.raises(RuntimeError, match="no explicit approved"):
        verify_approval(rejected, profile, policy, approval["bindings"])
    over = copy.deepcopy(profile); over["projections"]["paired_cell_hours"] = 20
    over["projections"]["conservative_paired_upper_bound_hours"] = 25
    assert evaluate_policy(over, policy)["passed"] is False


def test_review_and_require_cli_approve_exact_evidence_then_block_changed_profile(tmp_path):
    files, _, _, _ = approval_fixture(tmp_path)
    package_root = Path(__file__).resolve().parents[1]
    approval_path = tmp_path / "PROFILE_APPROVAL.json"
    common = ["--profile", str(files["profile"]), "--certification", str(files["cert"]),
        "--package-manifest", str(files["package"]), "--execution-plan", str(files["plan"]),
        "--cache-manifest", str(files["cache"]), "--resource-policy", str(files["policy"]),
        "--primary-slurm", str(files["slurm"])]
    environment = os.environ.copy(); environment["PYTHONPATH"] = str(package_root)
    review = subprocess.run([sys.executable, str(package_root / "scripts/review_profile.py"), *common,
        "--output", str(approval_path), "--reviewer", "fixture-reviewer", "--decision", "approve",
        "--reason", "all declared margins pass"], env=environment, text=True, capture_output=True)
    assert review.returncode == 0, review.stderr
    required = subprocess.run([sys.executable, str(package_root / "scripts/require_profile.py"), *common,
        "--approval", str(approval_path)], env=environment, text=True, capture_output=True)
    assert required.returncode == 0, required.stderr
    files["profile"].write_text(files["profile"].read_text() + "\n", encoding="utf-8")
    stale = subprocess.run([sys.executable, str(package_root / "scripts/require_profile.py"), *common,
        "--approval", str(approval_path)], env=environment, text=True, capture_output=True)
    assert stale.returncode != 0 and "stale, copied, or mismatched" in stale.stderr


def test_runbook_never_automatically_transitions_profile_to_primary():
    runbook = (Path(__file__).resolve().parents[1] / "DELFTBLUE_RUNBOOK.md").read_text(encoding="utf-8")
    assert "PROFILE_JOB=$(sbatch --parsable slurm/07_profile_100_batches.sbatch)" in runbook
    assert "review_profile.py" in runbook and "--decision approve" in runbook and "--decision reject" in runbook
    assert "--dependency=afterok:$PROFILE_JOB slurm/08_primary_fold0_seed0.sbatch" not in runbook
