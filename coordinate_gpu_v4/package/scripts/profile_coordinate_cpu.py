#!/usr/bin/env python3
"""Non-primary 100-optimizer-batch CPU profile for the frozen paired configuration."""
from __future__ import annotations
import argparse
import json
import math
import os
import resource
import time
from pathlib import Path
import numpy as np
import torch
from alignn.models.alignn import ALIGNN
from alignn_stage2.common import DATASET_SHA256, sha256_file, write_json
from alignn_stage2.cpu_coordinate_training import (BATCH_SIZE, EPOCHS, build_or_load_graphs,
    create_coordinate_start_states, forward_raw, graph_batch, iter_batches, load_inner_split, model_config, seed_all)
from alignn_stage2.cpu_runtime import assert_model_cpu, configure_cpu_runtime
from alignn_stage2.sigma_authorization import require_authorized_sigma

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--dataset",required=True); p.add_argument("--clean-cache-root",required=True)
    p.add_argument("--coordinate-cache-root",required=True); p.add_argument("--output",required=True); p.add_argument("--profile-work",required=True)
    a=p.parse_args(); authority=require_authorized_sigma(); runtime=configure_cpu_runtime(); dataset=Path(a.dataset)
    if sha256_file(dataset)!=DATASET_SHA256: raise RuntimeError("official dataset hash mismatch")
    train_ids, validation_ids, train_labels, validation_labels=load_inner_split(dataset,0,0)
    atoms, lines=build_or_load_graphs(dataset,train_ids,Path(a.clean_cache_root),"cpu_profile_inner_train")
    work=Path(a.profile_work); work.mkdir(parents=True,exist_ok=True)
    base,_=create_coordinate_start_states(0,0,work,Path(a.coordinate_cache_root),authority)
    seed_all(0); model=ALIGNN(model_config()).to(torch.device("cpu")); model.load_state_dict(base); assert_model_cpu(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    steps_per_epoch=math.ceil(len(train_ids)/BATCH_SIZE)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=1e-3,epochs=EPOCHS,steps_per_epoch=steps_per_epoch,
        pct_start=0.3,anneal_strategy="cos",cycle_momentum=True,div_factor=25,final_div_factor=10000)
    order=np.random.default_rng(0).permutation(len(train_ids)).tolist(); batches=list(iter_batches(order)); losses=[]
    started=time.perf_counter(); model.train()
    for step in range(100):
        indices=batches[step % len(batches)]; labels=torch.tensor([train_labels[i] for i in indices],dtype=torch.long,device="cpu")
        optimizer.zero_grad(set_to_none=True); logp,_=forward_raw(model,graph_batch(atoms,lines,indices),labels)
        loss=torch.nn.functional.nll_loss(logp,labels); loss.backward(); optimizer.step(); scheduler.step(); losses.append(float(loss.detach()))
    seconds=time.perf_counter()-started; package=Path(__file__).resolve().parents[1]
    report={"status":"passed","non_primary":True,"outer_test_accessed":False,"execution_device":"cpu",
        "measured_optimizer_batches":100,"finite_losses":bool(np.isfinite(losses).all()),"seconds":seconds,
        "seconds_per_batch":seconds/100,"projected_paired_40_epoch_hours":seconds/100*steps_per_epoch*EPOCHS*2/3600,
        "requested_wall_hours":24,"resume_required_if_projection_exceeds_walltime":True,
        "process_peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),"runtime":runtime,
        "coordinate_authorization_sha256":authority.authorization_sha256,
        "coordinate_cache_manifest_sha256":sha256_file(Path(a.coordinate_cache_root)/"fold_0"/"seed_0"/"COORDINATE_CACHE_MANIFEST.json"),
        "cpu_full_config_sha256":sha256_file(package/"CPU_FULL_CONFIG.json"),
        "profile_slurm_sha256":sha256_file(package/"slurm"/"07_profile_coordinate_100_batches_cpu.sbatch")}
    if not report["finite_losses"]: raise RuntimeError("profile losses are non-finite")
    write_json(Path(a.output),report); print("ALIGNN_COORDINATE_CPU_100_BATCH_PROFILE: PASS (manual approval required)"); return 0
if __name__ == "__main__": raise SystemExit(main())
