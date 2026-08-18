from __future__ import annotations
import ast
import json
import os
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]

def source(relative): return (ROOT/relative).read_text(encoding="utf-8-sig")

def test_release_sigma_is_deliberately_unresolved():
    value=json.loads(source("COORDINATE_NOISE_CONFIG.json"))
    assert value["sigma_cartesian_per_axis_angstrom"] is None
    assert value["sigma_status"]=="pending_prospective_selection"
    assert value["scientific_execution_authorized"] is False

def test_missing_and_test_only_sigma_boundaries(monkeypatch):
    from alignn_stage2.sigma_authorization import CoordinateSigmaAuthorizationMissing, require_authorized_sigma
    monkeypatch.delenv("ALIGNN_COORDINATE_SIGMA_AUTHORIZATION",raising=False)
    with pytest.raises(CoordinateSigmaAuthorizationMissing): require_authorized_sigma()
    monkeypatch.setenv("ALIGNN_COORDINATE_TEST_MODE","1")
    test={"sigma_cartesian_per_axis_angstrom":0.02,"test_only":True,"scientific_use_prohibited":True}
    assert require_authorized_sigma(test_config=test,allow_test_only=True).test_only
    with pytest.raises(CoordinateSigmaAuthorizationMissing):
        require_authorized_sigma(test_config={**test,"scientific_use_prohibited":False},allow_test_only=True)

def test_every_cpu_job_obeys_delftblue_policy():
    jobs=list((ROOT/"slurm").glob("*.sbatch")); assert len(jobs)==10
    prohibited=("--gres=gpu","--gpus","gpu-a100",".cuda()",'device="cuda"',"autocast(")
    for path in jobs:
        text=source(path.relative_to(ROOT)); lower=text.lower()
        for required in ("--account=research-ME-mse","--partition=compute","--ntasks=1",
                "--cpus-per-task=8","--mem-per-cpu=3968M",'export CUDA_VISIBLE_DEVICES=""'):
            assert required in text
        assert "--nodes" not in text
        assert all(token not in lower for token in prohibited)
    arrays=[source(path.relative_to(ROOT)) for path in jobs if "--array=" in source(path.relative_to(ROOT))]
    assert arrays and all("%5" in text for text in arrays)

def test_scientific_entry_points_gate_sigma_before_dataset_or_outer_access():
    paths=("alignn_stage2/coordinate_cache.py","alignn_stage2/cpu_coordinate_training.py",
        "alignn_stage2/cpu_coordinate_calibrate_export.py","alignn_stage2/cpu_coordinate_oof.py",
        "alignn_stage2/cpu_coordinate_final_audit.py")
    for path in paths:
        text=source(path); call=text.find("require_authorized_sigma(")
        assert call>=0
        later=[pos for token in ("sha256_file(dataset)","load_inner_split(","load_outer_test(","outer_test_predictions.csv")
            if (pos:=text.find(token))>=0]
        assert not later or call<min(later), path

def test_cpu_scientific_sources_have_no_accelerator_operations():
    for path in (ROOT/"alignn_stage2").glob("cpu_*.py"):
        text=path.read_text(encoding="utf-8").lower()
        if path.name=="cpu_runtime.py": continue
        assert ".cuda(" not in text and "torch.cuda." not in text and 'device="cuda"' not in text

def test_training_frozen_protocol_and_resume_contract_present():
    text=source("alignn_stage2/cpu_coordinate_training.py")
    for token in ("EPOCHS = 40","BATCH_SIZE = 32","AdamW","OneCycleLR","weight_decay=1e-5",
            "batch_position","rng_state","optimizer_steps","elapsed_runtime_seconds","INCOMPLETE_RESUME_REQUIRED.json","os.replace"):
        assert token in text
    assert "COMPLETE.json" not in text[text.find("def train_branch_cpu"):text.find("def train_branch(")]
    assert "fc.weight" in text and "fc.bias" in text and "optimizer_steps" in text

def test_coordinate_cache_is_separate_rebuilt_sharded_and_hashed():
    text=source("alignn_stage2/coordinate_cache.py")
    for token in ("Graph.atom_dgl_multigraph","COORDINATE_CACHE_MANIFEST.json","SHARD_SIZE = 256",
            "RECORD_COUNT = 3000","atom_graph_sha256","line_graph_sha256","authorization_sha256","os.replace"):
        assert token in text
    assert "clean cached graph" not in text.lower()

def test_prediction_schema_has_cpu_four_condition_contract():
    from alignn_stage2.cpu_prediction_schema import CONDITIONS, FIELDS, validate
    assert CONDITIONS=={"CPU_CONTROL_RAW","CPU_CONTROL_TS","CPU_RANDOM2_COORDINATE_RAW","CPU_RANDOM2_COORDINATE_TS"}
    assert {"structure_id","fold","seed","condition","split","true_label","raw_native_logit_0",
        "raw_native_logit_1","raw_probability_positive","predicted_label","checkpoint_sha256",
        "sample_order_index","execution_device"}.issubset(FIELDS)

def test_python_sources_parse():
    for path in ROOT.rglob("*.py"): ast.parse(path.read_text(encoding="utf-8-sig"),filename=str(path))
