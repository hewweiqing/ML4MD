"""Five-fold OOF consolidation and paired structure bootstrap."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .calibration_contract import DEFAULT_APPROVAL, _load_approved_module
from .common import FOLDS, SEEDS, sha256_ids, write_json
from .prediction_schema import read_csv
from .production import metrics as full_metrics


def binary_nll(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1))
    return -(shifted[np.arange(len(labels)), labels] - logsumexp)


def positive_probability(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities[:, 1]


def ece15(labels: np.ndarray, probabilities: np.ndarray) -> float:
    confidence = np.maximum(probabilities, 1 - probabilities)
    predictions = (probabilities >= 0.5).astype(int)
    edges, value = np.linspace(0, 1, 16), 0.0
    for index in range(15):
        selected = (confidence >= edges[index]) & (confidence < (edges[index + 1] if index < 14 else edges[index + 1] + 1e-12))
        if selected.any():
            value += selected.mean() * abs(np.mean(predictions[selected] == labels[selected]) - confidence[selected].mean())
    return float(value)


def summarize(logits: np.ndarray, labels: np.ndarray) -> dict:
    return full_metrics(labels, logits)


def paired_bootstrap(values: np.ndarray, repetitions: int, seed: int) -> dict:
    generator = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 50):
        count = min(50, repetitions - start)
        indices = generator.integers(0, len(values), size=(count, len(values)), endpoint=False)
        estimates[start:start + count] = values[indices].mean(axis=1)
    return {"estimate": float(values.mean()), "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)), "repetitions": repetitions, "seed": seed,
        "unit": "paired_structure"}


def clustered_structure_bootstrap(values_by_seed: np.ndarray, repetitions: int, seed: int) -> dict:
    """Resample structures once and carry every fixed seed/condition contrast together."""
    values = np.asarray(values_by_seed, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("clustered bootstrap requires a finite [seed,structure] contrast matrix")
    generator = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 50):
        count = min(50, repetitions - start)
        indices = generator.integers(0, values.shape[1], size=(count, values.shape[1]), endpoint=False)
        # Each sampled structure index selects all fixed seeds simultaneously.
        estimates[start:start + count] = values[:, indices].mean(axis=2).mean(axis=0)
    per_seed = values.mean(axis=1)
    return {"estimate": float(per_seed.mean()), "per_seed_estimates": per_seed.tolist(),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)), "repetitions": repetitions, "seed": seed,
        "unit": "structure_cluster_with_all_seed_predictions",
        "estimand": "mean across fixed seeds of each seed's five-fold OOF mean contrast",
        "uncertainty_scope": "structure-sampling CI conditional on the fixed Matbench folds and fixed five seeds"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), choices=SEEDS)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    args = parser.parse_args(argv)

    # Hash approval is checked before any outer-test prediction artifact is opened.
    _, approved_digest = _load_approved_module(DEFAULT_APPROVAL)
    work_root, output_dir = Path(args.work_root), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows, per_seed, comparisons_by_seed = [], {}, {}
    global_reference_ids = None
    for seed in args.seeds:
        ids_by_condition, labels_by_condition, logits_by_condition = {}, {}, {}
        seed_rows = []
        for fold in FOLDS:
            fold_root = work_root / f"fold_{fold}" / f"seed_{seed}"
            status = json.loads((fold_root / "CALIBRATION_EXPORT_STATUS.json").read_text(encoding="utf-8"))
            if status.get("approved_ts_sha256") != approved_digest:
                raise RuntimeError(f"fold {fold} seed {seed} used an unapproved TS module hash")
            rows = read_csv(fold_root / "predictions" / "outer_test_predictions.csv", expected_split="outer_test")
            sidecar = np.load(fold_root / "predictions" / "scaled_logits_sidecar.npz", allow_pickle=False)
            fold_ids = sidecar["outer_test_ids"].astype(str)
            fold_labels = sidecar["outer_test_labels"].astype(np.int64)
            condition_logits = {
                "A_Control_raw": np.asarray([[float(row["raw_native_logit_0"]), float(row["raw_native_logit_1"])] for row in rows if row["condition"] == "A_Control_raw"]),
                "B_Control_temperature_scaled": sidecar["outer_test_control"],
                "C_Random2_Descriptor_raw": np.asarray([[float(row["raw_native_logit_0"]), float(row["raw_native_logit_1"])] for row in rows if row["condition"] == "C_Random2_Descriptor_raw"]),
                "D_Random2_Descriptor_temperature_scaled": sidecar["outer_test_random2"],
            }
            for condition, logits in condition_logits.items():
                if logits.shape != (len(fold_ids), 2):
                    raise RuntimeError(f"fold {fold} seed {seed} condition {condition} shape mismatch")
                ids_by_condition.setdefault(condition, []).append(fold_ids)
                labels_by_condition.setdefault(condition, []).append(fold_labels)
                logits_by_condition.setdefault(condition, []).append(logits)
            seed_rows.extend(rows)
        reference_ids = np.concatenate(ids_by_condition["A_Control_raw"])
        reference_labels = np.concatenate(labels_by_condition["A_Control_raw"])
        if len(set(reference_ids.tolist())) != len(reference_ids):
            raise RuntimeError(f"seed {seed} OOF structures are not a one-time partition")
        if global_reference_ids is None:
            global_reference_ids = reference_ids.copy()
        elif not np.array_equal(reference_ids, global_reference_ids):
            raise RuntimeError("OOF structure identity/order differs across seeds; clustered resampling is unsafe")
        metrics, concatenated = {}, {}
        for condition in ids_by_condition:
            ids = np.concatenate(ids_by_condition[condition])
            labels = np.concatenate(labels_by_condition[condition])
            if not np.array_equal(ids, reference_ids) or not np.array_equal(labels, reference_labels):
                raise RuntimeError(f"seed {seed} condition alignment mismatch")
            concatenated[condition] = np.concatenate(logits_by_condition[condition])
            metrics[condition] = summarize(concatenated[condition], reference_labels)
        comparisons = {
            "B_minus_A_nll": binary_nll(concatenated["B_Control_temperature_scaled"], reference_labels) - binary_nll(concatenated["A_Control_raw"], reference_labels),
            "C_minus_A_nll": binary_nll(concatenated["C_Random2_Descriptor_raw"], reference_labels) - binary_nll(concatenated["A_Control_raw"], reference_labels),
            "D_minus_C_nll": binary_nll(concatenated["D_Random2_Descriptor_temperature_scaled"], reference_labels) - binary_nll(concatenated["C_Random2_Descriptor_raw"], reference_labels),
            "D_minus_B_nll": binary_nll(concatenated["D_Random2_Descriptor_temperature_scaled"], reference_labels) - binary_nll(concatenated["B_Control_temperature_scaled"], reference_labels),
        }
        for name, values in comparisons.items():
            comparisons_by_seed.setdefault(name, []).append(values)
        per_seed[str(seed)] = {"structure_count": len(reference_ids), "structure_ids_sha256": sha256_ids(reference_ids),
            "metrics": metrics, "paired_difference_estimates": {name: float(values.mean())
                for name, values in comparisons.items()},
            "primary_D_minus_B_nll": {"estimate": float(comparisons["D_minus_B_nll"].mean()),
                "definition": "NLL_D,s^OOF - NLL_B,s^OOF after concatenating the five official folds"}}
        all_rows.extend(seed_rows)
    if len({per_seed[str(seed)]["structure_ids_sha256"] for seed in args.seeds}) != 1:
        raise RuntimeError("OOF structure coverage differs across seeds")
    clustered = {name: clustered_structure_bootstrap(np.stack(values), args.bootstrap_repetitions,
        20260715) for name, values in comparisons_by_seed.items()}
    primary = clustered["D_minus_B_nll"]
    summary = {"status": "complete", "approved_ts_sha256": approved_digest, "folds": list(FOLDS), "seeds": args.seeds,
        "per_seed": per_seed, "matched_seed_mean_D_minus_B_nll": primary["estimate"],
        "primary_D_minus_B_nll": primary, "clustered_secondary_comparisons": clustered,
        "bootstrap_policy": "One structure resample carries all predictions for every fixed seed and A/B/C/D condition; "
            "the 5N seed-structure rows are never independent. Folds remain fixed and are not resampled.",
        "ci_interpretation": "structure-sampling CI conditional on the fixed Matbench folds; five seeds do not represent dataset-retraining uncertainty"}
    write_json(output_dir / "OOF_ANALYSIS.json", summary)
    with (output_dir / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
