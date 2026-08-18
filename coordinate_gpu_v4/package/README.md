# ALIGNN Stage-2 v29: CPU Random2-Coordinate

This additive package corrects v28's DGL 1.1.1 lazy-Column validation defect while retaining the complete CPU-only DelftBlue Random2-Coordinate experiment. Scientific execution remains fail-closed until a release-bound prospective sigma authority is imported.

Status: implementation complete; local validation recorded at packaging; DelftBlue runtime validation pending; sigma authorization pending; scientific results none.

Start with `DELFTBLUE_RUNBOOK.md`. The release configuration must retain `sigma_cartesian_per_axis_angstrom: null`. A separately generated, hash-bound prospective authority artifact is the only way to unlock coordinate cache construction and every later scientific entry point.

The experiment trains 50 CPU ALIGNN models (Control plus Random2-Coordinate for 25 fold/seed cells) and reports four readouts with validation-only MUBen temperature scaling. No GPU scientific path is provided by the v28 Slurm workflow.
