#!/usr/bin/env python3
"""Truthful phase-separated paired-experiment resource profile (non-primary)."""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from alignn.models.alignn import ALIGNN

from alignn_stage2.common import DATASET_SHA256, sha256_file, sha256_ids, write_json
from alignn_stage2.resource_profile import validate_profile
from alignn_stage2.structure_cache import MANIFEST_NAME, _SHARD_MEMORY_CACHE, canonical_digest
from alignn_stage2.training import (BATCH_SIZE, EPOCHS, build_or_load_graphs, graph_batch,
    component_state, iter_batches, load_inner_split, model_config, seed_all, states_byte_identical)


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def synchronized_timing(operation):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = operation()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def slurm_maxrss_evidence() -> dict:
    job_id = os.getenv("SLURM_JOB_ID")
    if not job_id:
        return {"available": False, "value": None, "reason": "SLURM_JOB_ID is unavailable; process ru_maxrss is reported"}
    completed = subprocess.run(["sstat", "-j", f"{job_id}.batch", "--format=MaxRSS", "--noheader"],
        text=True, capture_output=True, check=False)
    # sstat values are not stable while the batch step is still running, so
    # retain the raw evidence but use process RSS for the approval policy.
    return {"available": False, "value": None,
        "reason": "running-step sstat MaxRSS is not a reliable final peak; process ru_maxrss is used",
        "command_returncode": completed.returncode, "raw_stdout": completed.stdout.strip(),
        "raw_stderr": completed.stderr.strip()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--certification", required=True)
    parser.add_argument("--package-manifest", required=True)
    parser.add_argument("--execution-plan", required=True)
    parser.add_argument("--resource-policy", required=True)
    parser.add_argument("--primary-slurm", required=True)
    args = parser.parse_args()
    dataset, cache_root, output_path = Path(args.dataset), Path(args.cache_root), Path(args.output)
    certification, package_manifest = Path(args.certification), Path(args.package_manifest)
    execution_plan, resource_policy = Path(args.execution_plan), Path(args.resource_policy)
    primary_slurm = Path(args.primary_slurm)
    for path in (dataset, cache_root / MANIFEST_NAME, certification, package_manifest,
            execution_plan, resource_policy, primary_slurm):
        if not path.exists():
            raise RuntimeError(f"profile input is missing: {path}")
    if sha256_file(dataset) != DATASET_SHA256:
        raise RuntimeError("official dataset SHA-256 mismatch")
    certification_value = json.loads(certification.read_text(encoding="utf-8"))
    if certification_value.get("status") != "passed" or certification_value.get("a100_runtime_certified") is not True:
        raise RuntimeError("A100 runtime certification is required before profiling")
    package_value = json.loads(package_manifest.read_text(encoding="utf-8"))
    policy = json.loads(resource_policy.read_text(encoding="utf-8"))
    cache_manifest_path = cache_root / MANIFEST_NAME
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))

    train_ids, validation_ids, train_labels, validation_labels = load_inner_split(dataset, 0, 0)
    split_identity = canonical_digest({"fold": 0, "seed": 0, "train_ids_sha256": sha256_ids(train_ids),
        "validation_ids_sha256": sha256_ids(validation_ids)})

    # Phase 1 starts before any graph lookup, hash verification, or load.
    _SHARD_MEMORY_CACHE.clear()
    cache_phase_started = time.perf_counter()
    train_atoms, train_lines, train_cache = build_or_load_graphs(dataset, train_ids, cache_root,
        "profile_inner_train", return_stats=True)
    validation_atoms, validation_lines, validation_cache = build_or_load_graphs(dataset, validation_ids, cache_root,
        "profile_inner_validation", return_stats=True)
    cache_phase_seconds = time.perf_counter() - cache_phase_started
    cache_stats = {key: train_cache[key] + validation_cache[key] for key in (
        "requested_structures", "requested_shards", "manifest_lookup_seconds", "hash_verification_seconds",
        "graph_loading_seconds", "memory_cache_hits", "memory_cache_misses", "verified_shards")}
    cache_stats["total_verified_load_seconds"] = cache_phase_seconds
    if cache_stats["requested_structures"] != len(train_ids) + len(validation_ids):
        raise RuntimeError("cache request accounting mismatch")

    seed_all(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    init_started = time.perf_counter()
    model = ALIGNN(model_config()).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    steps_per_epoch = math.ceil(len(train_ids) / BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch, pct_start=0.3, anneal_strategy="cos", cycle_momentum=True,
        div_factor=25, final_div_factor=10000)
    torch.cuda.synchronize()
    init_seconds = time.perf_counter() - init_started

    order = np.random.default_rng(0).permutation(len(train_ids)).tolist()
    phase_totals = {name: 0.0 for name in ("graph_batch_cuda", "forward", "backward", "optimizer", "scheduler")}
    losses = []
    model.train()
    batch_window_started = time.perf_counter()
    for step in range(100):
        start = (step * BATCH_SIZE) % len(order)
        indices = order[start:start + BATCH_SIZE]
        if len(indices) < BATCH_SIZE:
            indices += order[:BATCH_SIZE - len(indices)]
        batch, elapsed = synchronized_timing(lambda: graph_batch(train_atoms, train_lines, indices))
        phase_totals["graph_batch_cuda"] += elapsed
        optimizer.zero_grad(set_to_none=True)
        labels = torch.tensor([train_labels[index] for index in indices], device="cuda")
        model_output, elapsed = synchronized_timing(lambda: model(batch))
        phase_totals["forward"] += elapsed
        loss = torch.nn.functional.nll_loss(model_output, labels)
        _, elapsed = synchronized_timing(loss.backward)
        phase_totals["backward"] += elapsed
        _, elapsed = synchronized_timing(optimizer.step)
        phase_totals["optimizer"] += elapsed
        _, elapsed = synchronized_timing(scheduler.step)
        phase_totals["scheduler"] += elapsed
        losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize()
    batch_window_seconds = time.perf_counter() - batch_window_started

    validation_started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for indices in iter_batches(list(range(len(validation_ids)))):
            model(graph_batch(validation_atoms, validation_lines, indices))
    torch.cuda.synchronize()
    validation_seconds = time.perf_counter() - validation_started

    descriptor_batch_limit = min(10, steps_per_epoch)
    descriptor_count = 0
    descriptor_samples = []
    descriptor_started = time.perf_counter()
    model.eval()
    encoder_before_descriptors = component_state(model, classifier=False)
    with torch.no_grad():
        for batch_number, indices in enumerate(iter_batches(list(range(len(train_ids))))):
            if batch_number >= descriptor_batch_limit:
                break
            captured = []
            hook = model.fc.register_forward_hook(lambda _module, inputs, _output: captured.append(inputs[0]))
            model(graph_batch(train_atoms, train_lines, indices))
            hook.remove()
            descriptor_batch = captured[-1]
            if descriptor_batch.ndim == 1:
                descriptor_batch = descriptor_batch.reshape(1, -1)
            descriptor_samples.append(descriptor_batch.detach())
            descriptor_count += len(indices)
    if not states_byte_identical(encoder_before_descriptors, component_state(model, classifier=False)):
        raise RuntimeError("profile descriptor extraction modified non-head parameters or BatchNorm buffers")
    torch.cuda.synchronize()
    descriptor_measured_seconds = time.perf_counter() - descriptor_started
    descriptor_projected_seconds = descriptor_measured_seconds * len(train_ids) / descriptor_count * 1.10

    descriptors = torch.cat(descriptor_samples)
    mean, std = descriptors.mean(0), descriptors.std(0, unbiased=False).clamp_min(1e-6)
    synthetic_count, warmup_steps = 3000, 938
    base_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    torch.cuda.synchronize()
    warmup_started = time.perf_counter()
    replay_states = []
    for replay in range(2):
        random2 = ALIGNN(model_config()).to("cuda")
        random2.load_state_dict(base_state)
        for name, parameter in random2.named_parameters():
            parameter.requires_grad_(name.startswith("fc."))
        generator = torch.Generator(device="cuda").manual_seed(10000)
        synthetic = mean + std * torch.randn((synthetic_count, descriptors.shape[1]), generator=generator, device="cuda")
        synthetic_labels = torch.tensor([0, 1] * (synthetic_count // 2), device="cuda")
        head_optimizer = torch.optim.AdamW(random2.fc.parameters(), lr=1e-4, weight_decay=0.0)
        cursor = 0
        permutation = torch.randperm(synthetic_count, generator=generator, device="cuda")
        for _ in range(warmup_steps):
            if cursor + 128 > synthetic_count:
                permutation = torch.randperm(synthetic_count, generator=generator, device="cuda")
                cursor = 0
            indices, cursor = permutation[cursor:cursor + 128], cursor + 128
            head_optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(random2.fc(synthetic[indices]), synthetic_labels[indices]).backward()
            head_optimizer.step()
        replay_states.append({name: value.detach().cpu().clone() for name, value in random2.state_dict().items()})
    torch.cuda.synchronize()
    warmup_replay_seconds = time.perf_counter() - warmup_started
    replay_deterministic = all(torch.equal(replay_states[0][name], replay_states[1][name]) for name in replay_states[0])
    if not replay_deterministic:
        raise RuntimeError("Random2 profile replay was not deterministic")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    current_output_size = directory_size(output_path.parent)
    with tempfile.TemporaryDirectory(prefix="profile_io_", dir=output_path.parent) as temporary:
        checkpoint_path = Path(temporary) / "representative_checkpoint.pt"
        io_started = time.perf_counter()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "history": [{"epoch": 0, "validation_nll": 1.0}]}, checkpoint_path)
        history_path = Path(temporary) / "history.json"
        write_json(history_path, [{"epoch": 0, "validation_nll": 1.0}])
        with checkpoint_path.open("rb") as stream:
            while stream.read(1024 * 1024):
                pass
        io_seconds = time.perf_counter() - io_started
        checkpoint_bytes = checkpoint_path.stat().st_size + history_path.stat().st_size

    branch_batch_seconds = batch_window_seconds / 100 * steps_per_epoch * EPOCHS
    validation_per_branch_seconds = validation_seconds * (EPOCHS + 2)
    io_per_branch_seconds = io_seconds * EPOCHS * 2
    control_seconds = init_seconds + branch_batch_seconds + validation_per_branch_seconds + io_per_branch_seconds
    random2_seconds = init_seconds + descriptor_projected_seconds + warmup_replay_seconds + branch_batch_seconds \
        + validation_per_branch_seconds + io_per_branch_seconds
    shared_start_state_init_seconds = init_seconds
    paired_seconds = cache_phase_seconds + shared_start_state_init_seconds + control_seconds + random2_seconds
    time_safety_factor = float(policy["profile"]["paired_time_safety_factor"])
    process_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    projected_host_bytes = int(process_rss_bytes * policy["profile"]["projected_host_memory_safety_factor"])
    gpu_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    cache_size = directory_size(cache_root)
    projected_cell_output = max(1, int(checkpoint_bytes * 8 * 1.5))
    cache_disk, output_disk = shutil.disk_usage(cache_root), shutil.disk_usage(output_path.parent)
    available_bytes = min(cache_disk.free, output_disk.free)
    projected_grid = cache_size + 25 * projected_cell_output + current_output_size

    report = {"schema_version": 2, "status": "passed", "release": "delftblue_package_v13",
        "non_primary": True, "fold": 0, "seed": 0, "package_aggregate_sha256": package_value["aggregate_sha256"],
        "dataset_sha256": DATASET_SHA256, "split_identity_sha256": split_identity,
        "cache_manifest_sha256": sha256_file(cache_manifest_path),
        "cache_provenance_sha256": cache_manifest["provenance"]["provenance_sha256"],
        "a100_runtime_certification_sha256": sha256_file(certification),
        "execution_plan_sha256": sha256_file(execution_plan), "resource_policy_sha256": sha256_file(resource_policy),
        "primary_slurm_sha256": sha256_file(primary_slurm),
        "requested_slurm_resources": policy["requested_slurm_resources"],
        "cache": cache_stats,
        "initialization": {"model_optimizer_scheduler_seconds": init_seconds,
            "shared_start_state_model_projected_seconds": shared_start_state_init_seconds},
        "training": {"measured_batches": 100, "finite_losses": bool(np.isfinite(losses).all()),
            "total_measured_seconds": batch_window_seconds, "seconds_per_batch": batch_window_seconds / 100,
            "graph_batch_cuda_seconds": phase_totals["graph_batch_cuda"], "forward_seconds": phase_totals["forward"],
            "backward_seconds": phase_totals["backward"], "optimizer_seconds": phase_totals["optimizer"],
            "scheduler_seconds": phase_totals["scheduler"], "steps_per_epoch": steps_per_epoch, "epochs": EPOCHS},
        "random2": {"descriptor_batches_measured": descriptor_batch_limit,
            "descriptor_structures_measured": descriptor_count, "descriptor_measured_seconds": descriptor_measured_seconds,
            "descriptor_projected_seconds": descriptor_projected_seconds, "descriptor_projection_safety_factor": 1.10,
            "warmup_steps_per_replay": warmup_steps, "replays": 2,
            "deterministic_replay_verified": replay_deterministic,
            "warmup_replay_measured_seconds": warmup_replay_seconds},
        "validation": {"representative_full_validation_seconds": validation_seconds,
            "validation_structures": len(validation_ids), "scheduled_evaluations_per_branch": EPOCHS + 2,
            "scheduled_evaluations_all_branches": 2 * (EPOCHS + 2)},
        "io": {"checkpoint_history_measured_seconds": io_seconds,
            "representative_checkpoint_history_bytes": checkpoint_bytes,
            "conservative_writes_per_epoch_per_branch": 2},
        "memory": {"peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "detected_gpu_total_bytes": int(gpu_properties.total_memory),
            "process_peak_rss_bytes": process_rss_bytes, "process_rss_source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            "projected_host_peak_bytes": projected_host_bytes,
            "projected_host_safety_factor": policy["profile"]["projected_host_memory_safety_factor"],
            "slurm_maxrss": slurm_maxrss_evidence()},
        "disk": {"cache_size_bytes": cache_size, "current_output_size_bytes": current_output_size,
            "projected_cell_output_bytes": projected_cell_output, "projected_full_grid_bytes": projected_grid,
            "available_bytes": available_bytes, "cache_filesystem_free_bytes": cache_disk.free,
            "output_filesystem_free_bytes": output_disk.free,
            "quota_evidence": "filesystem free-space measured locally; no reliable site quota API assumed"},
        "projections": {"control_branch_hours": control_seconds / 3600,
            "random2_branch_hours": random2_seconds / 3600, "paired_cell_hours": paired_seconds / 3600,
            "time_safety_factor": time_safety_factor,
            "conservative_paired_upper_bound_hours": paired_seconds / 3600 * time_safety_factor},
        "measured_phases": ["graph_cache_verified_load", "model_optimizer_scheduler_initialization",
            "100_optimizer_batches", "representative_full_inner_validation", "representative_descriptor_extraction",
            "random2_head_warmup_and_replay", "representative_checkpoint_history_io", "host_and_cuda_memory"],
        "excluded_phases": ["structure_graph_construction (cache must already exist)",
            "outer-test access and calibration", "Slurm queue time", "network transfer", "final OOF aggregation"],
        "calculation_formulas": {
            "branch_batch_seconds": "seconds_per_batch * ceil(train_count/batch_size) * epochs",
            "control_seconds": "model_init + branch_batches + full_validation*(epochs+2) + checkpoint_io*epochs*2",
            "random2_seconds": "model_init + descriptor_projection + measured_warmup_replay + branch_batches + full_validation*(epochs+2) + checkpoint_io*epochs*2",
            "paired_seconds": "verified_cache_load + shared_start_state_model_init + control_seconds + random2_seconds",
            "conservative_upper": "paired_seconds * paired_time_safety_factor",
            "projected_full_grid_disk": "cache_size + 25*projected_cell_output + current_output_size"}}
    if not report["training"]["finite_losses"]:
        report["status"] = "failed"
    validate_profile(report)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    write_json(temporary, report)
    os.replace(temporary, output_path)
    print("ALIGNN_100_BATCH_PROFILE: PASS (approval still required)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
