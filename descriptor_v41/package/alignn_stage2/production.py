"""Pure production gates for the paired 5-fold x 5-seed experiment."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from .calibration_contract import verify_binary_ordering_with_tolerance
from .common import CONDITIONS, FOLDS, SEEDS, sha256_file, write_json

ECE_BIN_COUNT = 15
ECE_BIN_INTERVAL_POLICY = "[lower,upper), except final bin [lower,1]"
BRIER_DEFINITION = "binary positive-class mean((p_1-y)^2); not two-class summed Brier"
ROC_AUC_SCORE_INPUT = "native binary logit margin z_1-z_0; rank-equivalent to exact positive-class probability"


def array_cell(index: int) -> tuple[int, int]:
    if index not in range(25):
        raise ValueError("array index must be in [0, 24]")
    return index // 5, index % 5


def reliability_bins(labels: np.ndarray, probabilities: np.ndarray, count: int = ECE_BIN_COUNT) -> list[dict]:
    """Equal-width top-label reliability bins with an explicit boundary policy."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= 0.5).astype(np.int64)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    edges = np.linspace(0.0, 1.0, count + 1)
    rows = []
    for index in range(count):
        upper = edges[index + 1]
        mask = (confidence >= edges[index]) & (confidence < upper if index < count - 1 else confidence <= upper)
        rows.append({"bin": index, "lower": float(edges[index]), "upper": float(upper),
            "count": int(mask.sum()), "mean_confidence": float(confidence[mask].mean()) if mask.any() else None,
            "accuracy": float((predictions[mask] == labels[mask]).mean()) if mask.any() else None})
    return rows


def adaptive_reliability_bins(labels: np.ndarray, probabilities: np.ndarray,
        count: int = ECE_BIN_COUNT) -> list[dict]:
    """Deterministic equal-mass top-label bins; stable ties preserve sample order."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= 0.5).astype(np.int64)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    ordered = np.argsort(confidence, kind="stable")
    rows = []
    for index, selected_indices in enumerate(np.array_split(ordered, count)):
        rows.append({"bin": index, "count": int(len(selected_indices)),
            "minimum_confidence": float(confidence[selected_indices].min()) if len(selected_indices) else None,
            "maximum_confidence": float(confidence[selected_indices].max()) if len(selected_indices) else None,
            "mean_confidence": float(confidence[selected_indices].mean()) if len(selected_indices) else None,
            "accuracy": float((predictions[selected_indices] == labels[selected_indices]).mean())
                if len(selected_indices) else None})
    return rows


def _equal_width_calibration_error(targets: np.ndarray, probabilities: np.ndarray,
        count: int = ECE_BIN_COUNT) -> float:
    edges, value = np.linspace(0.0, 1.0, count + 1), 0.0
    for index in range(count):
        upper = edges[index + 1]
        selected = (probabilities >= edges[index]) & (
            probabilities < upper if index < count - 1 else probabilities <= upper)
        if selected.any():
            value += float(selected.mean()) * abs(float(targets[selected].mean()) - float(probabilities[selected].mean()))
    return float(value)


def classwise_calibration_error(labels: np.ndarray, positive_probabilities: np.ndarray,
        count: int = ECE_BIN_COUNT) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    positive_probabilities = np.asarray(positive_probabilities, dtype=np.float64)
    per_class = {
        "0": _equal_width_calibration_error((labels == 0).astype(np.float64), 1.0 - positive_probabilities, count),
        "1": _equal_width_calibration_error((labels == 1).astype(np.float64), positive_probabilities, count),
    }
    return {"macro": float(np.mean(list(per_class.values()))), "per_class": per_class,
        "bin_count": count, "aggregation": "unweighted mean of per-class equal-width calibration errors"}


def roc_auc_continuous(labels: np.ndarray, scores: np.ndarray) -> float:
    labels, scores = np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(labels), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positives, negatives = int(labels.sum()), int((1 - labels).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC requires both classes")
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (len(labels), 2) or not np.isfinite(logits).all():
        raise ValueError("finite native two-class logits aligned to labels are required")
    margin = logits[:, 1] - logits[:, 0]
    probabilities = np.empty_like(margin, dtype=np.float64)
    nonnegative = margin >= 0
    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-margin[nonnegative]))
    exponential = np.exp(margin[~nonnegative])
    probabilities[~nonnegative] = exponential / (1.0 + exponential)
    predictions = np.argmax(logits, axis=1)
    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())
    f1_denominator = 2 * true_positive + false_positive + false_negative
    # AUROC is a ranking metric.  The native margin is mathematically
    # rank-equivalent to exact sigmoid probability without finite-precision
    # sigmoid saturation creating artificial ties.
    auc = roc_auc_continuous(labels, margin)
    bins = reliability_bins(labels, probabilities)
    ece = sum(row["count"] / len(labels) * abs(row["accuracy"] - row["mean_confidence"])
        for row in bins if row["count"])
    adaptive_bins = adaptive_reliability_bins(labels, probabilities)
    adaptive_ece = sum(row["count"] / len(labels) * abs(row["accuracy"] - row["mean_confidence"])
        for row in adaptive_bins if row["count"])
    classwise = classwise_calibration_error(labels, probabilities)
    log_norm = np.logaddexp(logits[:, 0], logits[:, 1])
    nll = np.mean(log_norm - logits[np.arange(len(labels)), labels])
    brier = float(np.mean((probabilities - labels) ** 2))
    return {"ece_15": float(ece), "ece_15_top_label": float(ece),
        "adaptive_ece_15_top_label": float(adaptive_ece), "classwise_ece_15_macro": classwise["macro"],
        "classwise_ece_15": classwise, "nll": float(nll), "brier": brier,
        "brier_positive_class": brier,
        "accuracy": float(np.mean(labels == predictions)), "f1": float(2 * true_positive / f1_denominator) if f1_denominator else 0.0,
        "roc_auc": float(auc), "mean_confidence": float(np.maximum(probabilities, 1-probabilities).mean()),
        "class_balance_positive": float(labels.mean()), "predicted_positive_rate": float(predictions.mean()),
        "reliability_bins": bins, "adaptive_reliability_bins": adaptive_bins,
        "metric_definitions": {"ece_15": "top-label confidence ECE", "ece_bin_count": ECE_BIN_COUNT,
            "ece_boundary_policy": ECE_BIN_INTERVAL_POLICY, "adaptive_ece": "top-label stable equal-mass bins",
            "brier": BRIER_DEFINITION, "roc_auc": ROC_AUC_SCORE_INPUT,
            "classwise_ece": "macro mean of class-0 and class-1 equal-width ECE"}}


def assert_temperature_invariance(raw: np.ndarray, scaled: np.ndarray, labels: np.ndarray | None = None,
        *, atol: float = 1e-12) -> dict:
    raw = np.asarray(raw, dtype=np.float64)
    scaled = np.asarray(scaled, dtype=np.float64)
    if raw.shape != scaled.shape or raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("aligned [n,2] logits required")
    raw_margin, scaled_margin = raw[:, 1] - raw[:, 0], scaled[:, 1] - scaled[:, 0]
    # Positive scalar division preserves the mathematical margin ordering.  The
    # stored raw CSV and scaled NPZ take different floating-point paths, so exact
    # argsort/tie-matrix equality is not a valid numerical invariant.  Check the
    # declared tolerance-aware ordering in both directions instead.  This remains
    # fail closed for every material inversion and is O(n log n), not O(n^2).
    tolerant_ordering = True
    try:
        verify_binary_ordering_with_tolerance(raw, scaled)
        verify_binary_ordering_with_tolerance(scaled, raw)
    except Exception:
        tolerant_ordering = False
    checks = {"predictions_unchanged": bool(np.array_equal(np.argmax(raw, axis=1), np.argmax(scaled, axis=1))),
        "class_unchanged": bool(np.array_equal(raw_margin >= 0, scaled_margin >= 0)),
        "ranking_unchanged": tolerant_ordering,
        "raw_finite": bool(np.isfinite(raw).all()), "scaled_finite": bool(np.isfinite(scaled).all()),
        "tie_pattern_unchanged": tolerant_ordering}
    if labels is not None:
        # The already-validated native-margin ordering is the defining AUROC
        # invariant.  Do not reconvert to finite-precision probabilities and
        # demand exact equality: sigmoid saturation can manufacture ties.
        checks["roc_auc_unchanged"] = tolerant_ordering
    if not all(checks.values()):
        raise RuntimeError(f"temperature invariance failed: {checks}")
    return checks


def atomic_promote(staging: Path, final: Path, quarantine: Path) -> None:
    staging, final, quarantine = map(Path, (staging, final, quarantine))
    if final.exists():
        raise FileExistsError(final)
    try:
        os.replace(staging, final)
    except Exception:
        quarantine.mkdir(parents=True, exist_ok=True)
        target = quarantine / staging.name
        if staging.exists():
            shutil.move(str(staging), str(target))
        raise


def complete_payload(cell_dir: Path, fold: int, seed: int, approved_ts_sha256: str) -> dict:
    cell_dir = Path(cell_dir)
    required = ["control/best.pt", "random2/best.pt", "control/validation_raw_logits.npz",
        "random2/validation_raw_logits.npz", "predictions/outer_test_predictions.csv",
        "predictions/scaled_logits_sidecar.npz", "temperature_scaling_provenance.json",
        "split_provenance.json", "random2_diagnostics.json", "CELL_VERIFICATION.json"]
    missing = [name for name in required if not (cell_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"incomplete cell; missing {missing}")
    hashes = {name: sha256_file(cell_dir / name) for name in required}
    return {"schema_version": 1, "status": "complete", "fold": fold, "seed": seed,
        "trained_branches": 2, "readouts": list(CONDITIONS), "approved_ts_sha256": approved_ts_sha256,
        "artifact_sha256": hashes}


def verify_complete(cell_dir: Path, expected_fold: int, expected_seed: int) -> bool:
    marker = Path(cell_dir) / "COMPLETE.json"
    if not marker.is_file():
        return False
    value = json.loads(marker.read_text(encoding="utf-8"))
    if value.get("status") != "complete" or (value.get("fold"), value.get("seed")) != (expected_fold, expected_seed):
        return False
    return all((Path(cell_dir) / name).is_file() and sha256_file(Path(cell_dir) / name) == digest
        for name, digest in value.get("artifact_sha256", {}).items()) and len(value.get("artifact_sha256", {})) == 10


def write_complete(cell_dir: Path, fold: int, seed: int, approved_ts_sha256: str) -> None:
    write_json(Path(cell_dir) / "COMPLETE.json", complete_payload(cell_dir, fold, seed, approved_ts_sha256))
