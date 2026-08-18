from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable

REQUIRED_COLUMNS = (
    "structure_id", "fold", "seed", "condition", "split", "true_label",
    "raw_native_logit_0", "raw_native_logit_1", "raw_probability_positive_audit_only",
    "predicted_label", "checkpoint_sha256", "sample_order_index",
)
ALLOWED_SPLITS = {"inner_validation", "outer_test"}
ALLOWED_CONDITIONS = {
    "A_Control_raw", "B_Control_temperature_scaled",
    "C_Random2_Descriptor_raw", "D_Random2_Descriptor_temperature_scaled",
}


class PredictionSchemaError(ValueError):
    pass


def stable_positive_probability(logit_0: float, logit_1: float) -> float:
    delta = float(logit_0) - float(logit_1)
    if delta >= 0:
        e = math.exp(-delta)
        return e / (1.0 + e)
    return 1.0 / (1.0 + math.exp(delta))


def validate_rows(rows: Iterable[dict[str, Any]], *, expected_split: str | None = None) -> list[dict[str, Any]]:
    materialized = list(rows)
    if not materialized:
        raise PredictionSchemaError("prediction artifact must contain at least one row")
    seen = set()
    for index, row in enumerate(materialized):
        missing = set(REQUIRED_COLUMNS) - set(row)
        if missing:
            raise PredictionSchemaError(f"row {index} missing columns: {sorted(missing)}")
        if row["split"] not in ALLOWED_SPLITS or (expected_split and row["split"] != expected_split):
            raise PredictionSchemaError(f"row {index} has invalid split")
        if row["condition"] not in ALLOWED_CONDITIONS:
            raise PredictionSchemaError(f"row {index} has invalid condition")
        if int(row["true_label"]) not in (0, 1) or int(row["predicted_label"]) not in (0, 1):
            raise PredictionSchemaError(f"row {index} has non-binary label")
        if int(row["sample_order_index"]) < 0:
            raise PredictionSchemaError(f"row {index} has negative sample-order index")
        if len(str(row["checkpoint_sha256"])) != 64:
            raise PredictionSchemaError(f"row {index} has invalid checkpoint hash")
        l0, l1 = float(row["raw_native_logit_0"]), float(row["raw_native_logit_1"])
        p = float(row["raw_probability_positive_audit_only"])
        if not all(math.isfinite(value) for value in (l0, l1, p)) or not 0.0 <= p <= 1.0:
            raise PredictionSchemaError(f"row {index} has invalid numeric value")
        expected_prediction = int(l1 > l0)
        if int(row["predicted_label"]) != expected_prediction:
            raise PredictionSchemaError(f"row {index} predicted label does not match native raw logits")
        if not math.isclose(p, stable_positive_probability(l0, l1), rel_tol=0.0, abs_tol=1e-12):
            raise PredictionSchemaError(f"row {index} audit probability is not derived from native raw logits")
        key = (row["structure_id"], int(row["fold"]), int(row["seed"]), row["condition"], row["split"])
        if key in seen:
            raise PredictionSchemaError(f"duplicate prediction key: {key}")
        seen.add(key)
    return materialized


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], *, expected_split: str | None = None) -> None:
    values = validate_rows(rows, expected_split=expected_split)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(values)
    temporary.replace(path)


def read_csv(path: str | Path, *, expected_split: str | None = None) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return validate_rows(rows, expected_split=expected_split)

