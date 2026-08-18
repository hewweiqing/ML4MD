"""CPU Random2-Coordinate OOF consolidation and paired structure bootstrap."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .calibration_contract import DEFAULT_APPROVAL, _load_approved_module
from .common import FOLDS, SEEDS, sha256_ids, write_json
from .gpu_prediction_schema import read_csv
from .production import metrics
from .sigma_authorization import require_authorized_sigma

CONDITIONS = (
    "GPU_CONTROL_RAW",
    "GPU_CONTROL_TS",
    "GPU_RANDOM2_COORDINATE_RAW",
    "GPU_RANDOM2_COORDINATE_TS",
)


def binary_nll(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    return np.logaddexp(logits[:, 0], logits[:, 1]) - logits[np.arange(len(labels)), labels]


def clustered_structure_bootstrap(values_by_seed: np.ndarray, repetitions: int = 5000,
        seed: int = 20260813) -> dict:
    """Resample structure columns, retaining every fixed seed and condition together."""
    values = np.asarray(values_by_seed, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 5 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("expected a finite [five seeds, structures] paired contrast matrix")
    generator = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 50):
        count = min(50, repetitions - start)
        sample = generator.integers(0, values.shape[1], size=(count, values.shape[1]))
        estimates[start:start + count] = values[:, sample].mean(axis=2).mean(axis=0)
    return {"estimate": float(values.mean()), "per_seed_estimates": values.mean(axis=1).tolist(),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)), "repetitions": repetitions,
        "seed": seed, "unit": "structure_cluster_with_all_five_seed_predictions"}


def _raw(rows: list[dict], condition: str) -> np.ndarray:
    selected = [row for row in rows if row["condition"] == condition]
    return np.asarray([[float(row["raw_native_logit_0"]), float(row["raw_native_logit_1"])]
        for row in selected], dtype=np.float64)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    args = parser.parse_args(argv)

    authorization = require_authorized_sigma()  # before opening any outer-test artifact
    _, ts_digest = _load_approved_module(DEFAULT_APPROVAL)
    work_root, output_dir = Path(args.work_root), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows, per_seed, contrasts = [], {}, {}
    global_ids = None
    for seed in SEEDS:
        ids_parts, label_parts = [], []
        logits_parts = {condition: [] for condition in CONDITIONS}
        seed_rows = []
        for fold in FOLDS:
            cell = work_root / f"fold_{fold}" / f"seed_{seed}"
            status = json.loads((cell / "CALIBRATION_EXPORT_STATUS.json").read_text(encoding="utf-8"))
            if status.get("approved_ts_sha256") != ts_digest:
                raise RuntimeError("unapproved temperature-scaling module hash")
            if status.get("coordinate_sigma_authorization_sha256") != authorization.authorization_sha256:
                raise RuntimeError("coordinate authorization differs from current authority")
            rows = read_csv(cell / "predictions" / "outer_test_predictions.csv", expected_split="outer_test")
            sidecar = np.load(cell / "predictions" / "scaled_logits_sidecar.npz", allow_pickle=False)
            ids = sidecar["outer_test_ids"].astype(str)
            labels = sidecar["outer_test_labels"].astype(np.int64)
            mapping = {
                CONDITIONS[0]: _raw(rows, CONDITIONS[0]),
                CONDITIONS[1]: np.asarray(sidecar["outer_test_control"], dtype=np.float64),
                CONDITIONS[2]: _raw(rows, CONDITIONS[2]),
                CONDITIONS[3]: np.asarray(sidecar["outer_test_random2"], dtype=np.float64),
            }
            for condition, logits in mapping.items():
                if logits.shape != (len(ids), 2):
                    raise RuntimeError(f"{fold}/{seed}/{condition}: logit shape mismatch")
                logits_parts[condition].append(logits)
            ids_parts.append(ids); label_parts.append(labels); seed_rows.extend(rows)
        ids = np.concatenate(ids_parts); labels = np.concatenate(label_parts)
        if len(set(ids.tolist())) != len(ids):
            raise RuntimeError("official folds are not a one-time OOF structure partition")
        if global_ids is None:
            global_ids = ids.copy()
        elif not np.array_equal(ids, global_ids):
            raise RuntimeError("OOF structure identity/order differs across seeds")
        combined = {key: np.concatenate(value) for key, value in logits_parts.items()}
        cell_contrasts = {
            "B_minus_A_nll": binary_nll(combined[CONDITIONS[1]], labels) - binary_nll(combined[CONDITIONS[0]], labels),
            "C_minus_A_nll": binary_nll(combined[CONDITIONS[2]], labels) - binary_nll(combined[CONDITIONS[0]], labels),
            "D_minus_C_nll": binary_nll(combined[CONDITIONS[3]], labels) - binary_nll(combined[CONDITIONS[2]], labels),
            "D_minus_B_nll": binary_nll(combined[CONDITIONS[3]], labels) - binary_nll(combined[CONDITIONS[1]], labels),
        }
        for name, value in cell_contrasts.items(): contrasts.setdefault(name, []).append(value)
        per_seed[str(seed)] = {"structure_count": len(ids), "structure_ids_sha256": sha256_ids(ids),
            "metrics": {key: metrics(labels, value) for key, value in combined.items()},
            "paired_difference_estimates": {key: float(value.mean()) for key, value in cell_contrasts.items()}}
        all_rows.extend(seed_rows)
    bootstraps = {key: clustered_structure_bootstrap(np.stack(value), args.bootstrap_repetitions)
        for key, value in contrasts.items()}
    result = {"status": "complete", "execution_device": "cuda", "folds": list(FOLDS), "seeds": list(SEEDS),
        "conditions": list(CONDITIONS), "approved_ts_sha256": ts_digest,
        "coordinate_authorization_sha256": authorization.authorization_sha256, "per_seed": per_seed,
        "primary_D_minus_B_nll": bootstraps["D_minus_B_nll"], "clustered_comparisons": bootstraps,
        "bootstrap_policy": "A sampled structure carries all five seeds and all four conditions; folds are fixed.",
        "smoke_and_profile_artifacts_included": False}
    write_json(output_dir / "GPU_COORDINATE_OOF_ANALYSIS.json", result)
    with (output_dir / "gpu_coordinate_oof_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_rows[0].keys()); writer.writeheader(); writer.writerows(all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
