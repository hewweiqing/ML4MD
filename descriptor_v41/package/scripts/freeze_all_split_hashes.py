#!/usr/bin/env python3
"""Freeze all 25 deterministic inner splits without materializing structures."""
from __future__ import annotations
import argparse, gzip, hashlib, json
from pathlib import Path
import ijson
from matbench.bench import MatbenchBenchmark
from sklearn.model_selection import train_test_split


def digest(values): return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()
def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset", required=True); parser.add_argument("--output", required=True); args=parser.parse_args()
    benchmark=MatbenchBenchmark(autoload=False, subset=["matbench_mp_is_metal"]); task=next(iter(benchmark.tasks))
    all_labels={}
    with gzip.open(args.dataset,"rb") as stream:
        row_index=-1
        for prefix,event,value in ijson.parse(stream):
            if prefix=="data.item.item" and event=="boolean":
                row_index+=1; all_labels[f"mb-mp-is-metal-{row_index+1:06d}"]=int(value)
    rows=[]
    for fold in range(5):
        fold_key=task.folds_map[fold]; outer_train=list(task.validation[fold_key].train); outer_test=list(task.validation[fold_key].test)
        allowed=set(outer_train); labels={sid: all_labels[sid] for sid in outer_train}
        if set(labels)!=allowed: raise RuntimeError("outer-training label coverage mismatch")
        for seed in range(5):
            train,val=train_test_split(outer_train,test_size=0.1,random_state=seed,shuffle=True,stratify=[labels[x] for x in outer_train])
            if set(train)&set(val) or (set(train)|set(val))!=allowed or set(outer_test)&allowed:
                raise RuntimeError("split overlap/coverage failure")
            rows.append({"fold":fold,"seed":seed,"inner_train_count":len(train),"inner_validation_count":len(val),
                "outer_test_count":len(outer_test),"inner_train_ids_sha256":digest(train),
                "inner_validation_ids_sha256":digest(val),"inner_validation_labels_sha256":digest(labels[x] for x in val),
                "outer_test_ids_sha256":digest(outer_test),"pairwise_disjoint":True,"complete_outer_train_coverage":True})
    value={"schema_version":1,"status":"frozen","dataset_sha256":file_digest(args.dataset),
        "split_method":"train_test_split(test_size=0.1,stratify=y,shuffle=True,random_state=seed)",
        "cells":rows,"cell_count":len(rows),"dataset_labels_streamed_once":len(all_labels),
        "fold_specific_outer_test_labels_used_for_inner_splits":False,"raw_labels_persisted":False}
    Path(args.output).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return 0
if __name__=="__main__": raise SystemExit(main())
