# DelftBlue v37 remaining-grid submission

v37 is an additive operational release. It does not alter the frozen model,
splits, seeds, intervention, calibration implementation, or analysis. It reuses
the v26 A100 certification and profile approval and the successful v36 primary
fold-0/seed-0 gate. Array index 0 is excluded because that cell is complete.

The training array covers indices 1 through 24. Calibration is submitted with
an `afterok` dependency and therefore starts only after every training array
task succeeds. OOF consolidation is deliberately not submitted automatically.

Before submission, define the v26 package, v36 package, verified v26 graph
cache, certified environment, and existing v26 staging/final/quarantine roots.
Use the commands supplied with the v37 handoff. Do not rerun preflight, graph
cache, smoke, profile, or fold-0/seed-0 training.
