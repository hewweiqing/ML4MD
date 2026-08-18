from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_oof_and_final_audit_use_external_immutable_gates_and_full_grid_audit():
    for name in ("30_consolidate_oof.sbatch", "40_final_audit.sbatch"):
        text = (ROOT / "slurm" / name).read_text(encoding="utf-8")
        assert "ALIGNN_V26_PACKAGE_ROOT" in text
        assert "ALIGNN_V36_PACKAGE_ROOT" in text
        assert "require_v37_external_gates.sh" in text
        assert "scripts/audit_grid_complete.py" in text
        assert "preflight/A100_RUNTIME_CERTIFICATION.json" not in text
        assert "require_profile_current.sh" not in text


def test_oof_scientific_analysis_contract_is_unchanged():
    text = (ROOT / "slurm/30_consolidate_oof.sbatch").read_text(encoding="utf-8")
    assert "--seeds 0 1 2 3 4" in text
    assert "--bootstrap-repetitions 5000" in text
    assert 'echo "ALIGNN_OOF_CONSOLIDATION: PASS"' in text
