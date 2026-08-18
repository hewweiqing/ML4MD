# CPU Prediction Artifact Schema

Every CSV row contains: `structure_id`, `fold`, `seed`, `condition`, `split`, `true_label`, `raw_native_logit_0`, `raw_native_logit_1`, `raw_probability_positive`, `scaled_probability_positive`, `predicted_label`, `temperature_reference_id`, `checkpoint_sha256`, `package_aggregate_sha256`, `coordinate_noise_config_sha256`, `sample_order_index`, and `execution_device` (`cpu`).

Conditions are `CPU_CONTROL_RAW`, `CPU_CONTROL_TS`, `CPU_RANDOM2_COORDINATE_RAW`, and `CPU_RANDOM2_COORDINATE_TS`. Splits are `inner_validation` and `outer_test`.

The native logits are the two values emitted by `ALIGNN.fc` before `LogSoftmax`. `raw_probability_positive` is audit-only and equals `softmax([z0,z1])[1] = sigmoid(z1-z0)`. The scaler consumes logits, never probabilities. One shared positive scalar divides both logits, so `softmax(z/T)[1] = sigmoid((z1-z0)/T)` within absolute tolerance `1e-12`. Positive scaling preserves predicted class and binary-logit ordering. Temperature references bind fitted metadata and approved MUBen source SHA-256.
