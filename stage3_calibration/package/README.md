# stage3_calibration: Random2 initialization diagnostic + Random2-Full warm-up

Adds two things missing from `descriptor_v41` (Random2-Descriptor) and
`coordinate_gpu_v4` (Random2-Coordinate): a measured (not assumed) check of
whether a randomly-initialized ALIGNN is actually overconfident at init
(Stage A — see `STAGE3_STAGE_A_PROTOCOL.md`), and a whole-network SCRATCH-
condition warm-up variant (Random2-Full) matching a randomly-initialized
backbone, since the existing head-only variants match the paper's
FINE-TUNING condition (appropriate only for a pretrained backbone).

Status: implementation complete for Stage A + the Random2-Full warm-up
primitive only. Stage B (training-set-size sweep) and Stage C (full 5-fold
run) are out of scope for this package as committed — see the plan history
for why (Stage A gates everything else, per its own protocol doc).

Start with `STAGE3_STAGE_A_PROTOCOL.md`, including its "Submission sequence"
section (`slurm/00...` preflight -> `slurm/05...` single-seed full-network
timing profile + required human approval -> `slurm/10...` full 20-seed run).
`alignn`/`dgl`/CUDA are required to run anything here; core mechanics have
since been dry-run end-to-end against real `alignn`/`dgl`/CUDA on a local
non-A100 GPU (catching and fixing two real bugs — see
`STAGE3_STAGE_A_PROTOCOL.md`), plus 32 passing local unit tests (`tests/`)
covering everything pure-torch. The full 20-seed run against the real
matbench dataset on an actual A100 has still never executed — that
requires DelftBlue.
