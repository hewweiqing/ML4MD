#!/usr/bin/env python3
"""Read-only calibration replay against existing completed MUBen artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


def max_metric_diff(left, right):
    values = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in set(left) & set(right):
            values.append(max_metric_diff(left[key], right[key]))
    else:
        try:
            a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
            if a.shape == b.shape and a.size:
                finite = np.isfinite(a) & np.isfinite(b)
                if finite.any():
                    values.append(float(np.max(np.abs(a[finite] - b[finite]))))
        except (TypeError, ValueError):
            pass
    return max(values, default=0.0)


def read_labels(path: Path):
    labels, masks = [], []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            labels.append(json.loads(row["labels"]))
            masks.append(json.loads(row["masks"]))
    return np.asarray(labels), np.asarray(masks)


def common_checks(raw_logits, scaled_logits, labels, masks, temperature):
    valid_rank = []
    for task in range(raw_logits.shape[1]):
        valid = masks[:, task].astype(bool)
        valid_rank.append(bool(np.array_equal(
            np.argsort(raw_logits[valid, task], kind="stable"),
            np.argsort(scaled_logits[valid, task], kind="stable"),
        )))
    return {
        "temperature_all_finite": bool(np.isfinite(temperature).all()),
        "temperature_all_positive": bool((np.asarray(temperature) > 0).all()),
        "raw_logits_unchanged": bool(np.array_equal(raw_logits, raw_logits.copy())),
        "labels_masks_shape_aligned": bool(raw_logits.shape == labels.shape == masks.shape),
        "ranking_preserved_per_task": valid_rank,
        "ranking_preserved_all_tasks": bool(all(valid_rank)),
    }


def replay_single(root: Path, family: str, seed: int):
    scripts = root / "reproduction" / "muben_ts" / "scripts"
    sys.path[:0] = [str(scripts), str(root / "MUBen")]
    from muben.utils.metrics import classification_metrics
    from muben.utils.io import load_results
    module = __import__("run_reproduction" if family == "DNN-RDKit" else "run_chemberta")
    hparams = module.load_hparams("primary")
    trainer, config, valid_dataset, test_dataset = module.build_trainer(
        "tox21", seed, hparams, "none", "primary"
    )
    raw_dir = root / "reproduction" / "muben_ts" / "results" / "muben_exact_reproduction" / "_native_outputs" / "primary"
    if family == "DNN-RDKit":
        raw_dir = raw_dir / "tox21" / "DNN-rdkit" / "none" / f"seed-{seed}"
        ts_dir = raw_dir.parents[1] / "TemperatureScaling" / f"seed-{seed}"
        result_dir = root / "reproduction" / "muben_ts" / "results" / "muben_exact_reproduction" / "tox21" / f"seed-{seed}" / "primary"
    else:
        raw_dir = raw_dir / "ChemBERTa" / "tox21" / "ChemBERTa" / "none" / f"seed-{seed}"
        ts_dir = raw_dir.parents[1] / "TemperatureScaling" / f"seed-{seed}"
        result_dir = root / "reproduction" / "muben_ts" / "results" / "muben_exact_reproduction" / "tox21" / f"seed-{seed}" / "primary" / "ChemBERTa"
    checkpoint = torch.load(raw_dir / "model_best.ckpt", map_location="cpu")
    trainer._model.load_state_dict(checkpoint["_state_dict"])
    trainer.model.eval()
    val_logits = np.asarray(trainer.inference(valid_dataset))
    test_logits = np.asarray(trainer.inference(test_dataset))
    val_labels, val_masks = np.asarray(valid_dataset.lbs), np.asarray(valid_dataset.masks)
    test_labels, test_masks = np.asarray(test_dataset.lbs), np.asarray(test_dataset.masks)
    # This is the exact completed-cell call path for DNN-RDKit/ChemBERTa.  It
    # consumes trainer.valid_dataset only; test logits/labels remain outside the call.
    module.set_seed(seed)
    trainer.ts_session()
    recomputed_t = trainer._ts_model.temperature.detach().cpu().numpy().copy()
    provenance = json.loads((result_dir / "ts_provenance.json").read_text(encoding="utf-8"))
    saved_t = np.asarray(provenance["temperature_check"]["per_task"])
    scaled_logits = test_logits / recomputed_t[None, :]
    raw_probs = 1.0 / (1.0 + np.exp(-test_logits))
    scaled_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
    saved_raw_probs, _, saved_labels, saved_masks = load_results([str(raw_dir / "preds" / "0.pt")])
    saved_scaled_probs, _, _, _ = load_results([str(ts_dir / "preds" / "0.pt")])
    raw_metrics = classification_metrics(raw_probs, test_labels, test_masks)
    scaled_metrics = classification_metrics(scaled_probs, test_labels, test_masks)
    saved_raw_metrics = json.loads((result_dir / "raw_metrics.json").read_text(encoding="utf-8"))
    saved_scaled_metrics = json.loads((result_dir / "ts_metrics.json").read_text(encoding="utf-8"))
    output = {
        "cell": {"model_family": family, "dataset": "tox21", "seed": seed, "config": "primary"},
        "artifact_mode": "raw checkpoint replay because validation logits were not persisted; calibration fitting only, no base-model training",
        "fit_inputs": ["validation_logits", "validation_labels", "validation_masks"],
        "test_labels_enter_fit": False,
        "saved_temperature": saved_t.tolist(),
        "recomputed_temperature": recomputed_t.tolist(),
        "temperature_max_abs_diff": float(np.max(np.abs(saved_t - recomputed_t))),
        "raw_probability_max_abs_diff_vs_saved": float(np.max(np.abs(raw_probs - saved_raw_probs))),
        "scaled_probability_max_abs_diff_vs_saved": float(np.max(np.abs(scaled_probs - saved_scaled_probs))),
        "saved_label_alignment": bool(np.array_equal(test_labels, saved_labels) and np.array_equal(test_masks, saved_masks)),
        "raw_metrics_max_abs_diff_vs_saved": max_metric_diff(raw_metrics, saved_raw_metrics),
        "scaled_metrics_max_abs_diff_vs_saved": max_metric_diff(scaled_metrics, saved_scaled_metrics),
    }
    output.update(common_checks(test_logits, scaled_logits, test_labels, test_masks, recomputed_t))
    return output


def replay_grover(root: Path, seed: int):
    scripts = root / "reproduction" / "muben_ts" / "scripts"
    sys.path[:0] = [str(scripts), str(root / "MUBen")]
    import grover_cached_logit_ts as scaling
    from muben.utils.metrics import classification_metrics
    cell = root / "reproduction" / "muben_ts" / "results" / "muben_exact_reproduction" / "tox21" / f"seed-{seed}" / "primary" / "GROVER"
    data = root / "MUBen" / "data" / "files" / "tox21"
    val_labels, val_masks = read_labels(data / "valid.csv")
    test_labels, test_masks = read_labels(data / "test.csv")
    val_atom, val_bond = np.load(cell / "val_atom_logits.npy"), np.load(cell / "val_bond_logits.npy")
    test_atom, test_bond = np.load(cell / "test_atom_logits.npy"), np.load(cell / "test_bond_logits.npy")
    fitted = scaling.fit_temperature_from_cached_logits(
        val_atom, val_bond, val_labels, val_masks, n_tasks=val_labels.shape[1], seed=seed
    )
    provenance = json.loads((cell / "ts_provenance.json").read_text(encoding="utf-8"))
    saved_atom = np.asarray(provenance["temperature_check"]["atom"]["per_task"])
    saved_bond = np.asarray(provenance["temperature_check"]["bond"]["per_task"])
    raw_probs = scaling.raw_probs_from_dual_logits(test_atom, test_bond)
    scaled_probs = scaling.apply_temperature_and_average(test_atom, test_bond, fitted.atom_temperature, fitted.bond_temperature)
    raw_metrics = classification_metrics(raw_probs, test_labels, test_masks)
    scaled_metrics = classification_metrics(scaled_probs, test_labels, test_masks)
    saved_raw_metrics = json.loads((cell / "raw_metrics.json").read_text(encoding="utf-8"))
    saved_scaled_metrics = json.loads((cell / "ts_metrics.json").read_text(encoding="utf-8"))
    checks_atom = common_checks(test_atom, test_atom / fitted.atom_temperature[None, :], test_labels, test_masks, fitted.atom_temperature)
    checks_bond = common_checks(test_bond, test_bond / fitted.bond_temperature[None, :], test_labels, test_masks, fitted.bond_temperature)
    return {
        "cell": {"model_family": "GROVER", "dataset": "tox21", "seed": seed, "config": "primary"},
        "artifact_mode": "persisted validation/test atom and bond logits",
        "fit_inputs": ["validation_atom_logits", "validation_bond_logits", "validation_labels", "validation_masks"],
        "test_labels_enter_fit": False,
        "atom_temperature_max_abs_diff": float(np.max(np.abs(saved_atom - fitted.atom_temperature))),
        "bond_temperature_max_abs_diff": float(np.max(np.abs(saved_bond - fitted.bond_temperature))),
        "raw_metrics_max_abs_diff_vs_saved": max_metric_diff(raw_metrics, saved_raw_metrics),
        "scaled_metrics_max_abs_diff_vs_saved": max_metric_diff(scaled_metrics, saved_scaled_metrics),
        "label_mask_alignment": bool(test_atom.shape == test_bond.shape == test_labels.shape == test_masks.shape),
        "atom_checks": checks_atom,
        "bond_checks": checks_bond,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muben-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.muben_root).resolve()
    results = []
    blockers = []
    for family in ("DNN-RDKit", "ChemBERTa"):
        try:
            results.append(replay_single(root, family, 0))
        except Exception as error:
            blockers.append({"cell": f"{family}/tox21/seed-0/primary", "error": f"{type(error).__name__}: {error}"})
    try:
        results.append(replay_grover(root, 0))
    except Exception as error:
        blockers.append({"cell": "GROVER/tox21/seed-0/primary", "error": f"{type(error).__name__}: {error}"})
    report = {"schema_version": 1, "status": "passed" if not blockers else "completed_with_blockers", "results": results, "blockers": blockers}
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
