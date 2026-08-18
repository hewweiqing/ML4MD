# DelftBlue v25 profiler-output correction

## Observed evidence

DelftBlue profile job `10633397` executed the representative 100 optimizer batches but failed before writing its report:

```text
AttributeError: 'Tensor' object has no attribute 'parent'
```

The profiler initially stored the requested report path in `output`, then reused `output` for the model tensor inside the batch loop. The final atomic-write preparation consequently called `.parent` on a tensor.

## Additive correction

v25 is copied from immutable v24. It changes only operational profiler naming:

- the report destination is named `output_path` throughout;
- the model result is named `model_output`;
- a regression test requires these roles to remain distinct;
- existing v24 login certification can be revalidated and rebound to the v25 aggregate.

No dependency, model setting, graph setting, dataset, fold, split, seed, Random2 behavior, MUBen temperature scaler, Slurm resource request, training behavior, calibration behavior, prediction schema, OOF calculation, or final-audit calculation changed.

The failed job produced no `FULL_CONFIG_100_BATCH_PROFILE.json`; therefore there is no partial profile eligible for approval. Primary execution remains fail-closed until v25 profiling completes and its exact evidence is manually reviewed and approved.
