# Prediction artifact schema

Every validation and outer-test prediction row is one structure/condition record. CSV is the canonical tabular format; UTF-8, comma delimiter, one header, and deterministic official split order are required.

| Column | Type | Contract |
|---|---|---|
| `structure_id` | string | Official Matbench structure identifier. |
| `fold` | integer | Official outer fold, 0 through 4. |
| `seed` | integer | Frozen seed, 0 through 4. |
| `condition` | enum | `A_Control_raw`, `B_Control_temperature_scaled`, `C_Random2_Descriptor_raw`, or `D_Random2_Descriptor_temperature_scaled`. |
| `split` | enum | `inner_validation` or `outer_test`. |
| `true_label` | integer | Binary label, 0 or 1. |
| `raw_native_logit_0` | float64 text | Native ALIGNN `fc(h)` logit for class 0, before `LogSoftmax`; never reconstructed from probability. |
| `raw_native_logit_1` | float64 text | Native ALIGNN `fc(h)` logit for class 1, before `LogSoftmax`; never reconstructed from probability. |
| `raw_probability_positive_audit_only` | float64 text | Stable softmax probability derived from the two raw logits; audit only and never supplied to temperature fitting. |
| `predicted_label` | integer | `argmax` of native raw logits. Positive scalar temperature scaling must not change it. |
| `checkpoint_sha256` | lowercase hex | SHA-256 of the selected branch checkpoint. |
| `sample_order_index` | integer | Zero-based position in the official split ordering. |

The scaled conditions retain both native raw-logit columns for lineage. Scaled logits used for metrics are stored in the calibrated artifact sidecar, not substituted into fields named `raw_native_logit_*`. The calibration sidecar records fitted metadata and the approved MUBen source digest.

Every branch `validation_raw_logits.npz` must additionally contain scalar `logit_source="ALIGNN.fc output before LogSoftmax"` and `logit_columns=["z_0","z_1"]`. Calibration fails closed if either value is absent or different. The executable forward contract verifies `model(batch) == log_softmax(z)` and `NLLLoss(log_softmax(z),y) == CrossEntropyLoss(z,y)` within `rtol=1e-6, atol=1e-7`; exported raw logits are never post-LogSoftmax values.

Uniqueness key: `(structure_id, fold, seed, condition, split)`. Within each `(fold, seed, condition, split)`, `sample_order_index` must be exactly `0..n-1`, and structure IDs and labels must align across all paired conditions.
