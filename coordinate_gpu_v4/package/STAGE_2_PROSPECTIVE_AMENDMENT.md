# Prospective Stage-2 protocol amendment

## Timing and status

This amendment was written on 2026-07-16 after the completed Stage 0B dataset/split audit and after the deliberately non-scientific Stage 1 smoke test. It was written before any full-fold training and before any outer-test structure, target, logit, prediction, metric, or evaluation was accessed. No Stage 1 metric or observed temperature was used to select or tune any choice below.

This amendment resolves the configuration blocker recorded in `STAGE_2_PROTOCOL_BLOCKER.md`; it does not delete, rewrite, or conceal that blocker. The amendment is additive and will be hashed before the full-configuration preflight starts.

## Provenance and reproduction boundary

Architecture and training settings were selected from the official Matbench ALIGNN implementation and then frozen as directed by the approved continuation prompt. Authoritative sources are:

- https://github.com/materialsproject/matbench/tree/main/benchmarks/matbench_v0.1_alignn
- https://matbench.materialsproject.org/Full%20Benchmark%20Data/matbench_v0.1_alignn/
- https://github.com/usnistgov/alignn

The historical Matbench submission reports ALIGNN 2021.12.27, DGL 0.6.1/DGL-CUDA 0.6.1 and PyTorch 1.10.1. This experiment instead uses pinned ALIGNN commit `f2366daa3413d28a825b46e34d001b5549b05a40`, ALIGNN package 2025.4.1, DGL 1.1.1+cu118 and PyTorch 2.0.1+cu118. It is therefore a current-pinned implementation using the historical architecture and is not an exact reproduction of the old leaderboard entry. The historical result is context only and will not be used for tuning.

## Native classification semantics verified before freezing

At the pinned commit, `ALIGNNConfig(classification=True, num_classes=2)` constructs the native `fc = Linear(hidden_features, 2)` and `LogSoftmax(dim=1)`. In `forward`, the precise authoritative temperature-scaling tensor is `out = self.fc(h)` after native graph pooling and immediately before `out = self.softmax(out)`. Its shape is `[batch, 2]`; it is called `raw_logits` in all Stage-2 artifacts. The native output is the two-class log-probability tensor and the pinned classification trainer selects `torch.nn.NLLLoss`. Stage 0A and Stage 1 runtime hooks already verified that `log_softmax(raw_logits)` exactly reconstructs the native returned log probabilities on CUDA. No external classification head is authorized.

## Frozen scientific protocol

The model, graph, supervised optimization, OneCycle, validation gates and execution rules are fully specified in the four companion JSON files. Official Matbench fold 0 and the completed Stage 0B seed-0 stratified split are immutable. Training, validation and outer test remain pairwise disjoint; the historical script's apparent duplication of training rows into validation is not reproduced.

Control and Random2-Descriptor start from one byte-identical native ALIGNN initialization. Random2 reuses `../RANDOM2_CONFIG.json` without changing its intervention, distribution, sample count, 938 steps, optimizer, learning rate, RNG offset or deterministic replay rule. Its warm-up updates only native `fc.weight` and `fc.bias`. The encoder must remain byte-identical, the classifier must change, and all held-out synthetic mechanical gates must pass before supervised training.

Both branches then run exactly 40 supervised epochs with identical epoch-level ordered sample-ID hashes. The supervised Python, NumPy, PyTorch CPU and CUDA RNGs are reset to seed 0 after Random2 warm-up. Fresh AdamW optimizers and OneCycleLR schedulers are created after that reset. The selected checkpoint is the epoch with minimum validation NLL, with earlier epoch winning an exact tie. There is no early termination.

Temperature scaling is fitted separately for each branch on selected-checkpoint validation raw logits only using `T=exp(t)` and unweighted NLL. Outer-test data may be evaluated exactly once only after all preceding integrity, Random2 and Control adequacy gates pass.

## Resource decision

The preflight must execute at least 100 representative full-configuration optimizer batches including graph construction/loading, native line graphs, CUDA transfer, forward, backward, optimizer update and per-update OneCycle step. If projected paired local runtime exceeds 12 hours, or disk/GPU/host-memory feasibility fails, local Stage 2 is prohibited and a DelftBlue package is produced without submission.

