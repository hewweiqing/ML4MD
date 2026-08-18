# GPU-coordinate v4 permutation-device correction

## Triggering evidence

DelftBlue smoke job `10654092` passed `REQUIRED_GPU_COORDINATE_GATES` and then failed before scientific smoke completion with:

```text
RuntimeError: Expected a 'cpu' device type for generator but found 'cuda'
```

The failure occurred while constructing the 3,000-record coordinate warm-up permutation. No outer-test access, calibration, primary training, or outcome inspection occurred.

## Additive correction

The original CPU experiment generates the warm-up permutation with a CPU `torch.Generator` seeded by `seed + 280000`. Version 4 retains that exact generator device, seed, `torch.randperm` calls, reshuffle boundary, batch size, and cursor behavior. Each selected CPU index batch is then copied to the active CUDA device solely for indexing the CUDA descriptor and label tensors.

This is a device-compatibility correction, not a scientific change. Generating the permutation directly with a CUDA generator was rejected because CPU and CUDA generators need not produce the same permutation stream for the same seed.

## Preservation

- Fixed coordinate dose: 0.040 Å per Cartesian axis.
- Official folds and seeds 0–4.
- Paired Control and Random2-Coordinate initialization.
- Exactly 938 warm-up steps, batch size 128, AdamW settings, and Random2 labels.
- Frozen model, graph, optimizer, scheduler, validation, calibration, and outer-test policy.
- Approved MUBen source hash.
- Immutable GPU-coordinate v1–v3 archives.

The regression test requires the CPU generator, forbids the failed CUDA-generator expression, verifies both permutation calls remain CPU-generated, and requires the explicit index transfer before CUDA tensor indexing.
