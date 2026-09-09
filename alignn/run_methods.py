"""ALIGNN: control versus fixed (F) and refreshed (R) random labels."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("DGLBACKEND", "pytorch")

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import random
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from alignn.graphs import Graph
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from jarvis.core.atoms import Atoms
from pymatgen.core import Element, Structure

from plot import metrics, plot

HERE = Path(__file__).resolve().parent
BATCH_SIZE, EVAL_BATCH_SIZE, LR = 8, 16, 0.001
ELEMENTS = [Element.from_Z(z).symbol for z in range(1, 101)]
MODEL = dict(name="alignn", alignn_layers=4, gcn_layers=4, atom_input_features=92,
             edge_input_features=80, triplet_input_features=40, embedding_features=64,
             hidden_features=256, output_features=2, link="identity", zero_inflated=False,
             classification=True, num_classes=2, extra_features=0)
GRAPH = dict(neighbor_strategy="k-nearest", cutoff=8.0, max_neighbors=12,
             atom_features="cgcnn", max_attempts=3, compute_line_graph=True,
             use_canonize=True, use_lattice_prop=False, cutoff_extra=3.5, dtype="float32")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def save_rows(path, rows):
    # Atomic replacement keeps progress tables readable during training.
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)


def new_model(seed, device):
    seed_all(seed)
    return ALIGNN(ALIGNNConfig(**MODEL)).to(device)


def snapshot(model):
    # Includes BatchNorm running statistics, not only trainable parameters.
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def cartesian(abc, matrix):
    return [sum(abc[i] * matrix[i][d] for i in range(3)) for d in range(3)]


def generate_bank(records, seed):
    """Original draw order: position attempts, species draws, then one label."""
    rng = random.Random(seed)
    bank, exhausted = [], 0
    for record in records:
        structure = copy.deepcopy(record["structure"])
        matrix, accepted, elements = structure["lattice"]["matrix"], [], []
        for site in structure["sites"]:
            best, best_distance = None, -1.0
            for _ in range(200):
                abc = [max(0.0, min(1.0, rng.gauss(0.5, 0.15) % 1.0)) for _ in range(3)]
                distances = []
                for other in accepted:
                    delta = [(a-b+0.5) % 1.0-0.5 for a, b in zip(abc, other)]
                    distances.append(sum(d**2 for d in cartesian(delta, matrix))**0.5)
                distance = min(distances) if distances else float("inf")
                if distance > best_distance:
                    best, best_distance = abc, distance
                if distance >= 0.5:
                    break
            else:
                exhausted += 1
            accepted.append(best)
            site["abc"], site["xyz"] = best, cartesian(best, matrix)
            site_elements = []
            for species in site["species"]:
                species["element"] = rng.choice(ELEMENTS)
                site_elements.append(species["element"])
            elements.extend(site_elements)
            if "label" in site:
                site["label"] = "/".join(site_elements)
        label = rng.choice([True, False])
        counts = {el: elements.count(el) for el in dict.fromkeys(elements)}
        bank.append(dict(mbid=record["mbid"], structure=structure, is_metal=label,
                         composition=" ".join(f"{el}{n}" for el, n in counts.items())))
    print(f"Bank seed {seed}: {len(bank)} structures; {exhausted} sites exhausted 200 attempts", flush=True)
    return bank


def build_items(records, synthetic=False):
    items = []
    for record in records:
        structure = record["structure"]
        if synthetic and len(structure["sites"]) > 1:
            # The original rejection sampler uses wrapped fractional deltas.
            # True periodic distances are checked independently, also for skew cells.
            dm = Structure.from_dict(structure).distance_matrix
            if dm[~np.eye(len(dm), dtype=bool)].min() < 0.5:
                raise ValueError(f"Synthetic distance below 0.5 Angstrom: {record['mbid']}; no records dropped")
        atoms = Atoms(lattice_mat=structure["lattice"]["matrix"],
                      coords=[site["xyz"] for site in structure["sites"]],
                      elements=[site["species"][0]["element"] for site in structure["sites"]],
                      cartesian=True)
        graph, line_graph = Graph.atom_dgl_multigraph(atoms, **GRAPH)
        items.append(dict(g=graph, lg=line_graph,
                          lattice=torch.tensor(atoms.lattice_mat).type(torch.get_default_dtype()),
                          label=int(record["is_metal"]), mbid=record["mbid"]))
        if len(items) % 200 == 0 or len(items) == len(records):
            print(f"Built {len(items)}/{len(records)} graphs", flush=True)
    return items


def batches(items, batch_size, device, label_gen=None):
    for start in range(0, len(items), batch_size):
        part = items[start:start+batch_size]
        labels = (torch.randint(0, 2, (len(part),), generator=label_gen) if label_gen is not None
                  else torch.tensor([it["label"] for it in part], dtype=torch.long))
        yield (dgl.batch([it["g"] for it in part]).to(device),
               dgl.batch([it["lg"] for it in part]).to(device),
               torch.stack([it["lattice"] for it in part]).to(device), labels.to(device))


def forward(model, graph, line_graph, lattice):
    return model((graph, line_graph, lattice)).reshape(-1, 2)


def check_contract(model, items, device):
    model.eval()
    graph, lg, lat, y = next(batches(items[:8], 8, device))
    logp = forward(model, graph, lg, lat)
    if not isinstance(model.softmax, torch.nn.LogSoftmax) or not torch.allclose(
            logp.logsumexp(1), torch.zeros_like(y, dtype=logp.dtype), atol=1e-5):
        raise ValueError("Expected native two-class log-probabilities")
    loss = F.nll_loss(logp, y)
    if not torch.allclose(loss, F.binary_cross_entropy_with_logits(logp[:, 1]-logp[:, 0], y.float()), atol=1e-6):
        raise ValueError("Classification NLL/BCE equivalence failed")
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not all(torch.isfinite(g).all() for g in gradients) or not any(g.abs().sum() > 0 for g in gradients):
        raise ValueError("Invalid classification gradients")
    model.zero_grad(set_to_none=True)


@torch.no_grad()
def predict(model, items, device):
    model.eval()
    outputs = [forward(model, g, lg, lat).cpu().numpy() for g, lg, lat, y in batches(items, EVAL_BATCH_SIZE, device)]
    logp = np.concatenate(outputs).astype(float)
    if not np.isfinite(logp).all():
        raise ValueError("Nonfinite predictions")
    return metrics(logp, np.asarray([it["label"] for it in items]))


@torch.no_grad()
def bank_nll(model, items, device, fresh=False):
    model.eval()
    total = 0.0
    for g, lg, lat, y in batches(items, EVAL_BATCH_SIZE, device):
        logp = forward(model, g, lg, lat)
        # R: integrate independent Bernoulli labels exactly, not a sampled task.
        total += float(-logp.mean(1).sum() if fresh else F.nll_loss(logp, y, reduction="sum"))
    if not math.isfinite(total):
        raise ValueError("Nonfinite bank NLL")
    return total / len(items)


def train_epoch(model, optimizer, items, rng, device, label_gen=None):
    order = list(range(len(items)))
    rng.shuffle(order)
    model.train()
    for g, lg, lat, y in batches([items[i] for i in order], BATCH_SIZE, device, label_gen):
        optimizer.zero_grad(set_to_none=True)
        loss = F.nll_loss(forward(model, g, lg, lat), y)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Nonfinite training gradient")
        optimizer.step()


def warmup(model, bank, independent, val, seed, arm, doses, device, rows, output):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    seed_all(300000+seed)
    order_rng = random.Random(400000+seed)
    labels = torch.Generator(device="cpu").manual_seed(500000+seed) if arm == "R" else None
    states = {}
    for epoch in range(max(doses)+1):
        if epoch:
            train_epoch(model, optimizer, bank, order_rng, device, labels)
        on = bank_nll(model, bank, device, fresh=arm == "R")
        off = bank_nll(model, independent, device, fresh=arm == "R")
        rows.append(dict(seed=seed, arm=arm, epoch=epoch, optimizer_updates=epoch*math.ceil(len(bank)/BATCH_SIZE),
                         on_bank_nll=on, off_bank_nll=off, memorization_gap=off-on,
                         evaluation_cohort="balanced_validation", **predict(model, val, device)))
        save_rows(output / "warmup.csv", rows)
        if epoch in doses:
            states[epoch] = snapshot(model)
            torch.save(states[epoch], output / "checkpoints" / f"seed_{seed}_{arm}_E{epoch}_warmup.pt")
        print(f"seed {seed} {arm} warm-up {epoch}/{max(doses)}: bank NLL {on:.4f}, independent {off:.4f}", flush=True)
    return states


def supervise(model, train, val, seed, arm, dose, epochs, device, rows, output):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    seed_all(600000+seed)
    order_rng = random.Random(700000+seed)
    for epoch in range(epochs+1):
        if epoch:
            train_epoch(model, optimizer, train, order_rng, device)
        row = dict(seed=seed, arm=arm, dose=dose, epoch=epoch,
                   evaluation_cohort="balanced_validation", **predict(model, val, device))
        rows.append(row)
        save_rows(output / "supervised.csv", rows)
        print(f"seed {seed} {arm} E={dose} supervised {epoch}/{epochs}: val NLL {row['nll']:.4f}", flush=True)
    torch.save(snapshot(model), output / "checkpoints" / f"seed_{seed}_{arm}_E{dose}_final.pt")


def run(args, doses):
    if (len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0
            or max(args.seeds) >= 2**31-700000 or len(set(doses)) != len(doses)
            or min(doses) < 1 or args.supervised_epochs < 1):
        raise ValueError("Use unique nonnegative seeds below 2**31-700000 and positive epoch counts")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new --output")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        try:
            dgl.graph(([0], [0])).to(device)
        except dgl.DGLError as error:
            raise RuntimeError("CUDA needs a GPU-enabled DGL build as well as PyTorch; see README") from error
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records, seen = {}, set()
    for split, count in (("train", 2014), ("val", 252), ("test", 252)):
        records[split] = json.loads((HERE / "data" / f"{split}.json").read_text(encoding="utf-8"))
        ids = {r["mbid"] for r in records[split]}
        if (len(ids) != count or len(records[split]) != count or seen & ids
                or any(type(r["is_metal"]) is not bool for r in records[split])
                or sum(r["is_metal"] for r in records[split])*2 != count):
            raise ValueError(f"Invalid frozen balanced {split} split")
        seen |= ids
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir()
    config = dict(seeds=args.seeds, doses=sorted(doses), supervised_epochs=args.supervised_epochs,
                  batch_size=BATCH_SIZE, eval_batch_size=EVAL_BATCH_SIZE, learning_rate=LR,
                  arms=["control", "F", "R"], device=str(device), model=MODEL, graph=GRAPH,
                  update_scope="full network", optimizer="Adam, reset at supervised handover",
                  seed_streams=dict(initial="seed", bank="100000+seed", independent="200000+seed",
                                    warmup="300000+seed", warmup_order="400000+seed",
                                    refreshed_labels="500000+seed", supervised="600000+seed", supervised_order="700000+seed"),
                  data_sha256={s: sha256(HERE / "data" / f"{s}.json") for s in records},
                  code_sha256={f: sha256(HERE / f) for f in ("run_methods.py", "run_doses.py", "plot.py")},
                  python=platform.python_version(), versions={p: importlib.metadata.version(p) for p in
                      ("torch", "dgl", "alignn", "jarvis-tools", "pymatgen", "numpy", "scipy", "pandas", "scikit-learn", "matplotlib")})
    save_json(output / "config.json", config)
    train, val = build_items(records["train"]), build_items(records["val"])
    warm_rows, supervised_rows = [], []
    branches = [("control", 0)] + [(arm, dose) for arm in ("F", "R") for dose in sorted(doses)]
    for seed in args.seeds:
        initial = new_model(seed, device)
        check_contract(initial, train, device)
        synthetic = []
        for role, offset in (("train", 100000), ("independent", 200000)):
            bank = generate_bank(records["train"], offset+seed)
            path = output / "banks" / f"seed_{seed}_{role}.json"
            save_json(path, bank)
            synthetic.append(build_items(bank, synthetic=True))
        supervise(copy.deepcopy(initial), train, val, seed, "control", 0, args.supervised_epochs,
                  device, supervised_rows, output)
        for arm in ("F", "R"):
            states = warmup(copy.deepcopy(initial), *synthetic, val, seed, arm, doses, device, warm_rows, output)
            for dose in sorted(doses):
                model = copy.deepcopy(initial)
                model.load_state_dict(states[dose])
                supervise(model, train, val, seed, arm, dose, args.supervised_epochs, device, supervised_rows, output)
                del model
            del states
        del initial, synthetic, bank
    # Test graphs/predictions are created only after ALL prescribed branches finish.
    save_json(output / "TRAINING_COMPLETE.json", dict(branches=len(args.seeds)*len(branches)))
    test, test_rows = build_items(records["test"]), []
    for seed in args.seeds:
        model = new_model(seed, device)
        for arm, dose in branches:
            state = torch.load(output / "checkpoints" / f"seed_{seed}_{arm}_E{dose}_final.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            test_rows.append(dict(seed=seed, arm=arm, dose=dose, **predict(model, test, device)))
    save_rows(output / "test.csv", test_rows)
    plot(output)
    save_json(output / "COMPLETE.json", dict(seeds=args.seeds, doses=sorted(doses)))
    print(f"Complete: {output}", flush=True)


def parser(description=__doc__, output="methods"):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--supervised-epochs", type=int, default=20)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--output", type=Path, default=HERE / "runs" / output)
    return p


if __name__ == "__main__":
    p = parser()
    p.add_argument("--warmup-epochs", type=int, default=4)
    args = p.parse_args()
    run(args, [args.warmup_epochs])
