# CPU Execution Contract

All v28 Slurm jobs use account `research-ME-mse`, partition `compute`, one task, eight CPUs, and `3968M` per CPU. Scientific jobs initially request 24 hours and arrays are capped at five concurrent tasks. No GPU resource is requested.

Every job exports `CUDA_VISIBLE_DEVICES=""`. Runtime code selects `torch.device("cpu")`; model parameters, graph tensors, logits, losses and optimizer state must remain on CPU. The runtime preflight rejects a visible accelerator. The static audit rejects GPU directives, accelerator device placement, `.cuda()`, autocast and accelerator-memory operations in v28 scientific paths.

The existing Python environment may contain a CUDA-enabled PyTorch build because it is the certified v26 environment. That frozen dependency provenance does not authorize or cause accelerator execution in v28.
