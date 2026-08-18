"""Validated-MUBen calibration and prediction export.

The approved scaler is loaded and both validation fits complete before this
module asks Matbench for outer-test IDs or reads any outer-test row.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from pathlib import Path

import ijson
import numpy as np
import torch
from alignn.models.alignn import ALIGNN
from matbench.bench import MatbenchBenchmark
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from .calibration_contract import apply_temperature, fit_temperature
from .common import DATASET_SHA256, FOLDS, SEEDS, append_jsonl, sha256_file, sha256_ids, write_json
from .gpu_prediction_schema import probability as stable_positive_probability, write_csv
from .gpu_coordinate_training import build_or_load_graphs, evaluate_raw, model_config, validate_frozen_files
from .gpu_runtime import DEVICE, assert_model_gpu, configure_gpu_runtime
from .sigma_authorization import require_authorized_sigma


def softmax_positive(logits: np.ndarray) -> np.ndarray:
    difference = logits[:, 0] - logits[:, 1]
    return np.where(difference >= 0, np.exp(-difference) / (1 + np.exp(-difference)), 1 / (1 + np.exp(difference)))


def validate_pair(control, random2) -> None:
    for key in ("structure_ids", "labels", "sample_order_index"):
        if not np.array_equal(control[key], random2[key]):
            raise RuntimeError(f"Control/Random2 validation alignment mismatch: {key}")
    if control["logits"].shape != (len(control["labels"]), 2) or random2["logits"].shape != control["logits"].shape:
        raise RuntimeError("validation native logits must have aligned shape [n, 2]")
    expected_source = "ALIGNN.fc output before LogSoftmax"
    expected_columns = np.asarray(["z_0", "z_1"])
    for branch_name, artifact in (("Control", control), ("Random2", random2)):
        if "logit_source" not in artifact.files or str(artifact["logit_source"].item()) != expected_source:
            raise RuntimeError(f"{branch_name} artifact does not prove pre-LogSoftmax ALIGNN.fc logit provenance")
        if "logit_columns" not in artifact.files or not np.array_equal(artifact["logit_columns"].astype(str), expected_columns):
            raise RuntimeError(f"{branch_name} artifact logit columns are not exactly z_0,z_1")


def control_adequacy(logits: np.ndarray, labels: np.ndarray, train_prior: float) -> dict:
    probabilities = softmax_positive(logits)
    predictions = np.argmax(logits, axis=1)
    nll = float(-np.mean(labels * np.log(np.clip(probabilities, 1e-15, 1)) + (1 - labels) * np.log(np.clip(1 - probabilities, 1e-15, 1))))
    brier = float(np.mean((probabilities - labels) ** 2))
    prior_nll = float(-np.mean(labels * np.log(train_prior) + (1 - labels) * np.log(1 - train_prior)))
    prior_brier = float(np.mean((train_prior - labels) ** 2))
    result = {"auroc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1_positive": float(f1_score(labels, predictions, pos_label=1)), "nll": nll, "brier": brier,
        "train_prior_nll": prior_nll, "train_prior_brier": prior_brier,
        "both_predicted_classes": len(set(predictions.tolist())) == 2,
        "finite_logits": bool(np.isfinite(logits).all()), "finite_probabilities": bool(np.isfinite(probabilities).all()),
        "positive_probability_std": float(np.std(probabilities)),
        "gate_scope": "Control basic classifier competence only",
        "outcome_neutral_with_respect_to_random2": True,
        "random2_comparison_used_for_gate": False}
    result["passed"] = bool(result["auroc"] >= 0.75 and result["balanced_accuracy"] >= 0.70 and result["f1_positive"] >= 0.65
        and prior_nll - nll >= 0.02 and prior_brier - brier >= 0.02 and result["both_predicted_classes"]
        and result["finite_logits"] and result["finite_probabilities"] and result["positive_probability_std"] >= 0.02)
    return result


def load_outer_test(dataset: Path, fold: int):
    benchmark = MatbenchBenchmark(autoload=False, subset=["matbench_mp_is_metal"])
    task = next(iter(benchmark.tasks))
    fold_key = task.folds_map[fold]
    test_ids = list(task.validation[fold_key].test)
    selected, labels = set(test_ids), {}
    with gzip.open(dataset, "rb") as stream:
        for row_index, row in enumerate(ijson.items(stream, "data.item", use_float=True)):
            structure_id = f"mb-mp-is-metal-{row_index + 1:06d}"
            if structure_id in selected:
                labels[structure_id] = int(row[1])
    if set(labels) != selected:
        raise RuntimeError("outer-test ID/label coverage mismatch")
    return test_ids, [labels[item] for item in test_ids]


def load_selected_model(checkpoint: Path):
    model = ALIGNN(model_config()).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE)["model"])
    assert_model_gpu(model)
    return model


def rows_for(ids, labels, raw_logits, scaled_logits, fold, seed, condition, split, checkpoint_hash,
        temperature_reference_id, package_hash, coordinate_config_hash):
    return [{"structure_id": structure_id, "fold": fold, "seed": seed, "condition": condition, "split": split,
        "true_label": int(labels[index]), "raw_native_logit_0": float(raw_logits[index, 0]),
        "raw_native_logit_1": float(raw_logits[index, 1]),
        "raw_probability_positive": stable_positive_probability(*raw_logits[index]),
        "scaled_probability_positive": stable_positive_probability(*scaled_logits[index]),
        "predicted_label": int(raw_logits[index, 1] > raw_logits[index, 0]), "checkpoint_sha256": checkpoint_hash,
        "temperature_reference_id": temperature_reference_id, "package_aggregate_sha256": package_hash,
        "coordinate_noise_config_sha256": coordinate_config_hash,
        "sample_order_index": index, "execution_device": "cuda"} for index, structure_id in enumerate(ids)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=FOLDS)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    args = parser.parse_args(argv)
    authorization = require_authorized_sigma()
    configure_gpu_runtime()
    validate_frozen_files()
    dataset = Path(args.dataset)
    if sha256_file(dataset) != DATASET_SHA256:
        raise RuntimeError("official dataset SHA-256 mismatch")
    work = Path(args.work_root) / f"fold_{args.fold}" / f"seed_{args.seed}"
    control_path, random2_path = work / "control" / "validation_raw_logits.npz", work / "random2_coordinate" / "validation_raw_logits.npz"
    if not control_path.is_file() or not random2_path.is_file():
        raise RuntimeError("both completed branch validation-logit artifacts are required")
    control, random2 = np.load(control_path, allow_pickle=False), np.load(random2_path, allow_pickle=False)
    validate_pair(control, random2)
    split_provenance = json.loads((work / "split_provenance.json").read_text(encoding="utf-8"))
    control_checkpoint_provenance = json.loads((work / "control" / "checkpoint_provenance.json").read_text(encoding="utf-8"))
    if sha256_ids(control["structure_ids"]) != split_provenance["inner_validation_ids_sha256"] or sha256_ids(control["labels"]) != split_provenance["inner_validation_labels_sha256"]:
        raise RuntimeError("validation raw-logit ID/label provenance mismatch")
    if control_checkpoint_provenance["checkpoint_reconstruction_max_abs_diff"] > 1e-7:
        raise RuntimeError("selected Control checkpoint reconstruction gate failed")
    adequacy = control_adequacy(control["logits"], control["labels"], split_provenance["inner_train_positive_fraction"])
    write_json(work / "validation_adequacy.json", adequacy)
    if not adequacy["passed"]:
        raise RuntimeError("frozen Control validation adequacy gate failed")

    # Fail-closed boundary: both fits and applications occur before any outer-test split lookup.
    fitted_control = fit_temperature(control["logits"], control["labels"], split="validation")
    fitted_random2 = fit_temperature(random2["logits"], random2["labels"], split="validation")
    scaled_control_validation = apply_temperature(control["logits"], fitted_control)
    scaled_random2_validation = apply_temperature(random2["logits"], fitted_random2)
    calibration = {"control": fitted_control.public_metadata(), "random2": fitted_random2.public_metadata(),
        "fit_scope": "inner_validation_only", "input": "native binary raw logits",
        "native_logit_source": "ALIGNN.fc output before LogSoftmax", "native_logit_columns": ["z_0", "z_1"],
        "prohibited_input": "post-LogSoftmax log-probabilities or reconstructed probabilities"}
    write_json(work / "temperature_scaling_provenance.json", calibration)

    sentinel = work / "OUTER_TEST_ACCESS_STARTED.json"
    if sentinel.exists():
        raise RuntimeError("outer-test access previously started; refusing automatic repeated materialization")
    write_json(sentinel, {"fold": args.fold, "seed": args.seed, "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "time": time.time(), "approved_ts_sha256": fitted_control.source_sha256,
        "policy": "single materialization attempt; interruption requires manual adjudication"})
    append_jsonl(work / "RUN_MANIFEST_HISTORY.jsonl", {"event": "OUTER_TEST_ACCESS_STARTED", "fold": args.fold,
        "seed": args.seed, "time": time.time(), "approved_ts_sha256": fitted_control.source_sha256})
    test_ids, test_labels = load_outer_test(dataset, args.fold)
    cache_root = Path(os.environ["ALIGNN_GRAPH_CACHE_ROOT"])
    test_atoms, test_lines = build_or_load_graphs(dataset, test_ids, cache_root, "outer_test")
    control_checkpoint, random2_checkpoint = work / "control" / "best.pt", work / "random2_coordinate" / "best.pt"
    control_hash, random2_hash = sha256_file(control_checkpoint), sha256_file(random2_checkpoint)
    control_test = evaluate_raw(load_selected_model(control_checkpoint), test_atoms, test_lines, test_labels)
    random2_test = evaluate_raw(load_selected_model(random2_checkpoint), test_atoms, test_lines, test_labels)
    scaled_control_test, scaled_random2_test = apply_temperature(control_test, fitted_control), apply_temperature(random2_test, fitted_random2)
    output = work / "predictions"
    package_hash = json.loads((Path(__file__).resolve().parents[1] / "PACKAGE_MANIFEST.json").read_text())["aggregate_sha256"]
    coordinate_config_hash = sha256_file(Path(__file__).resolve().parents[1] / "COORDINATE_NOISE_CONFIG.json")
    control_ref, random2_ref = f"control:{fitted_control.source_sha256}:T={fitted_control.temperature}", f"coordinate:{fitted_random2.source_sha256}:T={fitted_random2.temperature}"
    validation_rows = []
    validation_rows += rows_for(control["structure_ids"], control["labels"], control["logits"], control["logits"], args.fold, args.seed, "GPU_CONTROL_RAW", "inner_validation", control_hash, "raw:T=1", package_hash, coordinate_config_hash)
    validation_rows += rows_for(control["structure_ids"], control["labels"], control["logits"], scaled_control_validation, args.fold, args.seed, "GPU_CONTROL_TS", "inner_validation", control_hash, control_ref, package_hash, coordinate_config_hash)
    validation_rows += rows_for(random2["structure_ids"], random2["labels"], random2["logits"], random2["logits"], args.fold, args.seed, "GPU_RANDOM2_COORDINATE_RAW", "inner_validation", random2_hash, "raw:T=1", package_hash, coordinate_config_hash)
    validation_rows += rows_for(random2["structure_ids"], random2["labels"], random2["logits"], scaled_random2_validation, args.fold, args.seed, "GPU_RANDOM2_COORDINATE_TS", "inner_validation", random2_hash, random2_ref, package_hash, coordinate_config_hash)
    test_rows = []
    test_rows += rows_for(test_ids, test_labels, control_test, control_test, args.fold, args.seed, "GPU_CONTROL_RAW", "outer_test", control_hash, "raw:T=1", package_hash, coordinate_config_hash)
    test_rows += rows_for(test_ids, test_labels, control_test, scaled_control_test, args.fold, args.seed, "GPU_CONTROL_TS", "outer_test", control_hash, control_ref, package_hash, coordinate_config_hash)
    test_rows += rows_for(test_ids, test_labels, random2_test, random2_test, args.fold, args.seed, "GPU_RANDOM2_COORDINATE_RAW", "outer_test", random2_hash, "raw:T=1", package_hash, coordinate_config_hash)
    test_rows += rows_for(test_ids, test_labels, random2_test, scaled_random2_test, args.fold, args.seed, "GPU_RANDOM2_COORDINATE_TS", "outer_test", random2_hash, random2_ref, package_hash, coordinate_config_hash)
    write_csv(output / "inner_validation_predictions.csv", validation_rows, expected_split="inner_validation")
    write_csv(output / "outer_test_predictions.csv", test_rows, expected_split="outer_test")
    np.savez_compressed(output / "scaled_logits_sidecar.npz", validation_control=scaled_control_validation,
        validation_random2=scaled_random2_validation, outer_test_control=scaled_control_test,
        outer_test_random2=scaled_random2_test, outer_test_ids=np.asarray(test_ids), outer_test_labels=np.asarray(test_labels))
    write_json(work / "CALIBRATION_EXPORT_STATUS.json", {"status": "complete", "fold": args.fold, "seed": args.seed,
        "outer_test_evaluated_for_metrics": False, "outer_test_predictions_exported": True,
        "outer_test_ids_sha256": sha256_ids(test_ids), "approved_ts_sha256": fitted_control.source_sha256,
        "coordinate_sigma_authorization_sha256": authorization.authorization_sha256,
        "native_logit_source": "ALIGNN.fc output before LogSoftmax", "native_logit_columns": ["z_0", "z_1"]})
    write_json(work / "COMPLETE.json", {"status": "complete", "fold": args.fold, "seed": args.seed,
        "execution_device": "cuda", "trained_models": ["control", "random2_coordinate"],
        "reported_conditions": ["GPU_CONTROL_RAW", "GPU_CONTROL_TS", "GPU_RANDOM2_COORDINATE_RAW",
            "GPU_RANDOM2_COORDINATE_TS"], "coordinate_sigma_authorization_sha256": authorization.authorization_sha256,
        "approved_ts_sha256": fitted_control.source_sha256, "control_checkpoint_sha256": control_hash,
        "random2_coordinate_checkpoint_sha256": random2_hash,
        "outer_test_metrics_computed_in_cell": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

