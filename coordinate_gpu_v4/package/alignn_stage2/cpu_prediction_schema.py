from __future__ import annotations

import csv
import math
from pathlib import Path


CONDITIONS = {"CPU_CONTROL_RAW", "CPU_CONTROL_TS", "CPU_RANDOM2_COORDINATE_RAW", "CPU_RANDOM2_COORDINATE_TS"}
SPLITS = {"inner_validation", "outer_test"}
FIELDS = ("structure_id", "fold", "seed", "condition", "split", "true_label",
    "raw_native_logit_0", "raw_native_logit_1", "raw_probability_positive",
    "scaled_probability_positive", "predicted_label", "temperature_reference_id",
    "checkpoint_sha256", "package_aggregate_sha256", "coordinate_noise_config_sha256",
    "sample_order_index", "execution_device")


def probability(z0: float, z1: float) -> float:
    margin = float(z1) - float(z0)
    return 1 / (1 + math.exp(-margin)) if margin >= 0 else math.exp(margin) / (1 + math.exp(margin))


def validate(rows, expected_split=None):
    rows = list(rows); seen = set()
    if not rows: raise ValueError("empty prediction artifact")
    for row in rows:
        if set(row) != set(FIELDS): raise ValueError("prediction schema mismatch")
        if row["condition"] not in CONDITIONS or row["split"] not in SPLITS or expected_split and row["split"] != expected_split:
            raise ValueError("invalid condition/split")
        if row["execution_device"] != "cpu": raise ValueError("CPU prediction artifact has non-CPU device")
        if len(str(row["checkpoint_sha256"])) != 64 or len(str(row["package_aggregate_sha256"])) != 64:
            raise ValueError("invalid artifact hash")
        z0, z1 = float(row["raw_native_logit_0"]), float(row["raw_native_logit_1"])
        if not math.isclose(float(row["raw_probability_positive"]), probability(z0, z1), abs_tol=1e-12):
            raise ValueError("raw probability is not derived from two native logits")
        if int(row["predicted_label"]) != int(z1 > z0): raise ValueError("predicted label mismatch")
        key = tuple(row[x] for x in ("structure_id", "fold", "seed", "condition", "split"))
        if key in seen: raise ValueError("duplicate prediction row")
        seen.add(key)
    return rows


def write_csv(path, rows, expected_split=None):
    rows = validate(rows, expected_split); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def read_csv(path, expected_split=None):
    with Path(path).open(encoding="utf-8", newline="") as stream: rows = list(csv.DictReader(stream))
    return validate(rows, expected_split)
