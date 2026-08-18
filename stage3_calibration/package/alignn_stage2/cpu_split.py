"""Frozen official-fold and deterministic inner-validation definitions (shared with descriptor_v41/coordinate_gpu_v4)."""
from __future__ import annotations
import gzip, json
from pathlib import Path
import ijson
from matbench.bench import MatbenchBenchmark
from sklearn.model_selection import train_test_split
from .common import sha256_ids

PACKAGE_ROOT=Path(__file__).resolve().parents[1]
GRAPH_SETTINGS={"neighbor_strategy":"k-nearest","cutoff":8.0,"max_neighbors":12,"atom_features":"cgcnn","compute_line_graph":True,"use_canonize":True}

def load_inner_split(dataset: Path, fold: int, seed: int):
    benchmark=MatbenchBenchmark(autoload=False,subset=["matbench_mp_is_metal"]); task=next(iter(benchmark.tasks))
    outer_training_ids=list(task.validation[task.folds_map[fold]].train); allowed=set(outer_training_ids); labels={}
    with gzip.open(dataset,"rb") as stream:
        row_index=-1
        for prefix,event,value in ijson.parse(stream):
            if prefix=="data.item.item" and event=="boolean":
                row_index+=1; structure_id=f"mb-mp-is-metal-{row_index+1:06d}"
                if structure_id in allowed: labels[structure_id]=int(value)
    if set(labels)!=allowed: raise RuntimeError("outer-training ID/label coverage mismatch")
    train_ids,validation_ids=train_test_split(outer_training_ids,test_size=0.1,random_state=seed,shuffle=True,
        stratify=[labels[item] for item in outer_training_ids])
    audit=json.loads((PACKAGE_ROOT/"ALL_25_SPLIT_HASHES.json").read_text(encoding="utf-8"))
    expected=next(item for item in audit["cells"] if item["fold"]==fold and item["seed"]==seed)
    if (sha256_ids(train_ids)!=expected["inner_train_ids_sha256"] or sha256_ids(validation_ids)!=expected["inner_validation_ids_sha256"]
            or sha256_ids(labels[item] for item in validation_ids)!=expected["inner_validation_labels_sha256"]):
        raise RuntimeError("frozen fold/seed split hash mismatch")
    return train_ids,validation_ids,[labels[item] for item in train_ids],[labels[item] for item in validation_ids]
