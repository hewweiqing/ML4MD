"""Scratch ORB: control and three input/label refresh policies, four warm-up passes."""
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import random
from dataclasses import dataclass
from pathlib import Path

import ase
from ase.data import chemical_symbols
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.models import segment_ops
from orb_models.forcefield import pretrained
from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter

from plot import ARMS, metrics, plot

HERE = Path(__file__).resolve().parent
TRAIN_LIMITS, EVAL_LIMITS = (8, 128, 24000), (16, 256, 30000)
MODEL = dict(latent_dim=256, base_mlp_hidden_dim=1024, base_mlp_depth=2,
             head_mlp_hidden_dim=256, head_mlp_depth=1, num_message_passing_steps=5,
             activation="silu", has_charge_spin_cond=False, has_stress=True, device="cpu")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_rows(path, rows):
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Classifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(256, 2)

    def forward(self, graph):
        features = self.backbone(graph)["node_features"]
        pooled = segment_ops.aggregate_nodes(tensor=features, n_node=graph.n_node, reduction="mean")
        return self.classifier(pooled)


def new_model(seed):
    seed_all(seed)
    # This factory initializes random weights; it does not download pretrained weights.
    # Retain its construction order (including discarded forcefield heads) for RNG parity.
    native = pretrained.orb_v3_direct_architecture(**MODEL)
    return Classifier(native.model)


def snapshot(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def to_atoms(structure):
    return ase.Atoms(symbols=[s["species"][0]["element"] for s in structure["sites"]],
                     positions=np.asarray([s["xyz"] for s in structure["sites"]], dtype=np.float64),
                     cell=np.asarray(structure["lattice"]["matrix"], dtype=np.float64), pbc=True)


def cartesian(abc, matrix):
    return [sum(abc[i]*matrix[i][d] for i in range(3)) for d in range(3)]


def raw_bank(clean, seed):
    """Amendment-001 draw order and metadata, independent of any old source file."""
    rng, rows = random.Random(seed), []
    for record in clean:
        structure = copy.deepcopy(record["structure"])
        matrix, accepted, positions, elements = structure["lattice"]["matrix"], [], [], []
        for site in structure["sites"]:
            original, best, best_distance = list(site["abc"]), None, -1.0
            for attempts in range(1, 201):
                abc = [max(0.0, min(1.0, rng.gauss(.5, .15) % 1.0)) for _ in range(3)]
                distances = []
                for other in accepted:
                    delta = [(a-b+.5) % 1.0-.5 for a, b in zip(abc, other)]
                    distances.append(math.sqrt(sum(d*d for d in cartesian(delta, matrix))))
                distance = min(distances) if distances else math.inf
                if distance > best_distance:
                    best, best_distance = abc, distance
                if distance >= .5:
                    break
            accepted.append(best)
            site["abc"], site["xyz"] = best, cartesian(best, matrix)
            positions.append(dict(orig_abc=original, new_abc=best, attempts=attempts,
                                  achieved_min_dist=None if math.isinf(best_distance) else best_distance))
            site_elements = []
            for species in site["species"]:
                old, new = species["element"], rng.choice(chemical_symbols[1:101])
                species["element"] = new
                site_elements.append(new)
                elements.append(dict(orig_element=old, new_element=new))
            if "label" in site:
                site["label"] = "/".join(site_elements)
        label = rng.choice([True, False])
        counts = {}
        for entry in elements:
            el = entry["new_element"]
            counts[el] = counts.get(el, 0)+1
        rows.append(dict(mbid=record["mbid"], composition=" ".join(f"{el}{n}" for el, n in counts.items()),
                         orig_composition=record["composition"], site_element_map=elements,
                         position_noise_std=.15, position_min_dist_floor=.5, site_position_map=positions,
                         is_metal=label, orig_is_metal=record["is_metal"], structure=structure))
    return rows


def invalid_geometry(clean, rows):
    bad = set()
    for source, row in zip(clean, rows, strict=True):
        a, b = source["structure"], row["structure"]
        if (source["mbid"] != row["mbid"] or source["is_metal"] != row["orig_is_metal"]
                or len(a["sites"]) != len(b["sites"])
                or not np.allclose(a["lattice"]["matrix"], b["lattice"]["matrix"], atol=1e-8, rtol=0)):
            raise ValueError("Synthetic bank changed source identity, lattice or site count")
        atoms = to_atoms(b)
        if atoms.numbers.max() > 100 or not np.isfinite(atoms.positions).all():
            raise ValueError("Unsupported species or nonfinite geometry")
        minimum = atoms.get_all_distances(mic=True)[np.triu_indices(len(atoms), 1)].min() if len(atoms)>1 else math.inf
        if minimum < .5-1e-10 or any(s["achieved_min_dist"] is not None and s["achieved_min_dist"] < .5
                                    for s in row["site_position_map"]):
            bad.add(row["mbid"])
    return bad


def generate_bank(clean, seed, epoch):
    bank_seed = 100000+seed if epoch == 1 else stable_seed("epoch_bank", seed, epoch)
    rows, repairs = raw_bank(clean, bank_seed), []
    for attempt in range(21):
        bad = invalid_geometry(clean, rows)
        if not bad:
            return dict(seed=seed, epoch=epoch, generator_seed=bank_seed, records=rows,
                        geometry_passed=True, deterministic_geometry_repairs=repairs)
        if attempt == 20:
            raise ValueError(f"Geometry still invalid after 20 repair rounds: {sorted(bad)}; floor not relaxed")
        for i, record in enumerate(clean):
            if record["mbid"] in bad:
                retry = stable_seed("geometry_retry", bank_seed, record["mbid"], attempt)
                rows[i] = raw_bank([record], retry)[0]
                repairs.append(dict(mbid=record["mbid"], retry=attempt, seed=retry))


@dataclass
class Item:
    graph: AtomGraphs
    label: int
    mbid: str
    n_atoms: int
    n_edges: int


def build_items(records):
    adapter = ForcefieldAtomsAdapter(radius=6., max_num_neighbors=120)
    items = []
    for record in records:
        atoms = to_atoms(record["structure"])
        graph = adapter.from_ase_atoms(atoms, device="cpu")
        items.append(Item(graph, int(record["is_metal"]), str(record["mbid"]), len(atoms), int(graph.n_edge.sum().item())))
        if len(items) % 200 == 0 or len(items) == len(records):
            print(f"Built {len(items)}/{len(records)} graphs", flush=True)
    return items


def index_batches(items, limits=TRAIN_LIMITS, seed=None):
    order = list(range(len(items)))
    if seed is not None:
        random.Random(seed).shuffle(order)
    batch, atoms, edges = [], 0, 0
    max_items, max_atoms, max_edges = limits
    for i in order:
        item = items[i]
        if item.n_edges > max_edges:
            raise ValueError(f"{item.mbid}: {item.n_edges} edges exceed hard limit {max_edges}")
        if batch and (len(batch) >= max_items or atoms+item.n_atoms > max_atoms or edges+item.n_edges > max_edges):
            yield batch
            batch, atoms, edges = [], 0, 0
        batch.append(i)
        atoms, edges = atoms+item.n_atoms, edges+item.n_edges
    if batch:
        yield batch  # Original oversized-atom singleton policy; edge limit remains hard.


def paired_batches(banks, seed, epoch):
    worst = [copy.copy(item) for item in banks[0]]
    for i, item in enumerate(worst):
        if any(bank[i].mbid != item.mbid or bank[i].n_atoms != item.n_atoms for bank in banks):
            raise ValueError("Unpaired bank identities/site counts")
        item.n_edges = max(bank[i].n_edges for bank in banks)
    return list(index_batches(worst, seed=400000+seed+epoch))


class Banks:
    """Methods: maximum over all banks. Doses: epoch-local fixed/current maximum."""
    def __init__(self, clean, seed, epochs, output, batching):
        self.clean, self.seed, self.output, self.batching = clean, seed, output, batching
        self.current_epoch, self.current = 1, self.load(1)
        self.fixed = self.current
        self.all = [self.fixed]+[self.load(e) for e in range(2, epochs+1)] if batching == "all_banks" else None

    def load(self, epoch):
        path = self.output / "banks" / f"seed_{self.seed}" / f"epoch_{epoch:02d}.json"
        if not path.exists():
            save_json(path, generate_bank(self.clean, self.seed, epoch))
        payload = read_json(path)
        if payload["seed"] != self.seed or payload["epoch"] != epoch or not payload["geometry_passed"]:
            raise ValueError("Cached bank identity/audit mismatch")
        print(f"seed {self.seed}: bank epoch {epoch}", flush=True)
        return build_items(payload["records"])

    def get(self, epoch):
        if self.all is not None:
            return self.all[epoch-1]
        if epoch != self.current_epoch:
            self.current = None
            gc.collect()
            self.current = self.fixed if epoch == 1 else self.load(epoch)
            self.current_epoch = epoch
        return self.current

    def batches(self, epoch):
        return paired_batches(self.all if self.all is not None else [self.fixed, self.get(epoch)], self.seed, epoch)


def collate(items, indices, device):
    if str(device).startswith("cuda") and torch.cuda.memory_reserved() > 3.5*2**30:
        torch.cuda.empty_cache()
    return (AtomGraphs.batch([items[i].graph for i in indices]).to(device=device),
            torch.tensor([items[i].label for i in indices], dtype=torch.long, device=device))


def release_cache(device):
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def train_epoch(model, optimizer, items, batches, device, labels_rng=None):
    model.train()
    order, labels_audit, total = [], [], 0.
    for indices in batches:
        graph, labels = collate(items, indices, device)
        if labels_rng is not None:
            labels = torch.randint(0, 2, (len(indices),), generator=labels_rng).to(device)
        order.append([items[i].mbid for i in indices])
        labels_audit.extend(labels.detach().cpu().tolist())
        optimizer.zero_grad(set_to_none=True)
        logits = model(graph)
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach())*len(indices)
        del graph, labels, logits, loss
    if sorted(i for batch in batches for i in batch) != list(range(len(items))):
        raise ValueError("Epoch did not visit every record once")
    release_cache(device)
    return dict(loss=total/len(items), examples=len(items), updates=len(batches),
                order_sha256=digest(order), labels_sha256=digest(labels_audit), positive_labels=sum(labels_audit))


@torch.no_grad()
def evaluate(model, items, device):
    model.eval()
    outputs = []
    for indices in index_batches(items, EVAL_LIMITS):
        graph, labels = collate(items, indices, device)
        logits = model(graph)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite predictions")
        outputs.append(logits.detach().cpu())
        del graph, labels, logits
    logits = torch.cat(outputs).double()
    labels = torch.tensor([it.label for it in items], dtype=torch.long)
    # Preserve native double-precision Torch softmax/CE; plot.py itself needs no Torch.
    result = metrics(logits.numpy(), labels.numpy(), torch.softmax(logits, 1).numpy(), float(F.cross_entropy(logits, labels)))
    release_cache(device)
    return result


def check_contract(initial, train, device):
    model = copy.deepcopy(initial).to(device).eval()
    graph, labels = collate(train, next(index_batches(train)), device)
    logits = model(graph)
    ce = F.cross_entropy(logits, labels)
    bce = F.binary_cross_entropy_with_logits(logits[:, 1]-logits[:, 0], labels.float())
    if logits.shape != (len(labels), 2) or not torch.allclose(ce, bce, atol=1e-6):
        raise ValueError("Native two-logit classification contract failed")
    ce.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not all(torch.isfinite(g).all() for g in gradients) or not any(g.abs().sum() > 0 for g in gradients):
        raise ValueError("Nonfinite/zero classification gradients")
    del model, graph, logits, ce, bce
    release_cache(device)


def phase(initial, train, val, banks, seed, arm, doses, supervised_epochs, device, output, rows, dose=0):
    warm = banks is not None
    model = copy.deepcopy(initial).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0. if warm else 1e-5)
    labels_rng = torch.Generator(device="cpu").manual_seed(500000+seed)
    seed_all((300000 if warm else 600000)+seed)
    epochs, cumulative, states = max(doses) if warm else supervised_epochs, 0, {}
    name = "warmup" if warm else "supervised"
    for epoch in range(epochs+1):
        extra = dict(loss=0., examples=0, updates=0, order_sha256="", labels_sha256="", positive_labels=0)
        if epoch:
            batches = banks.batches(epoch) if warm else list(index_batches(train, seed=700000+seed+epoch))
            items = (banks.get(epoch) if arm == ARMS[3] else banks.fixed) if warm else train
            fresh = labels_rng if warm and arm != ARMS[1] else None
            extra = train_epoch(model, optimizer, items, batches, device, fresh)
            del items, batches
        cumulative += extra["updates"]
        rows.append(dict(seed=seed, arm=arm, dose=dose, phase=name, epoch=epoch, cumulative_updates=cumulative,
                         **evaluate(model, val, device), **extra))
        save_rows(output / f"{name}.csv", rows)
        if warm and epoch in doses:
            path = output / "checkpoints" / f"seed_{seed}_{arm}_W{epoch:03d}.pt"
            torch.save(snapshot(model), path)
            states[epoch] = path
        print(f"seed {seed} {arm} {name} E={dose} {epoch}/{epochs}: confidence={rows[-1]['mean_confidence']:.4f} accuracy={rows[-1]['accuracy']:.4f}", flush=True)
    if not warm:
        torch.save(snapshot(model), output / "checkpoints" / f"seed_{seed}_{arm}_W{dose:03d}_final.pt")
    del model, optimizer
    release_cache(device)
    return states


def check_pairing(warm, supervised):
    for field in ("order_sha256", "updates"):
        if warm.groupby(["seed", "epoch"])[field].nunique().max() != 1:
            raise ValueError("Warm-up batch pairing failed")
    if warm[warm.arm.isin(ARMS[2:])].groupby(["seed", "epoch"]).labels_sha256.nunique().max() != 1:
        raise ValueError("Fresh-label pairing failed")
    if supervised.groupby(["seed", "epoch"]).order_sha256.nunique().max() != 1:
        raise ValueError("Supervised batch pairing failed")


def run(args, doses, batching):
    if (len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0 or max(args.seeds) >= 2**32-700000-max(doses)-args.supervised_epochs
            or len(set(doses)) != len(doses) or min(doses) < 1 or args.supervised_epochs < 1):
        raise ValueError("Use unique nonnegative seeds and positive, unique doses/epoch counts")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new --output")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    clean, validation = (read_json(HERE / "data" / f"{s}.json") for s in ("train", "val"))
    seen = set()
    for records, count in ((clean, 2014), (validation, 252)):
        ids = {r["mbid"] for r in records}
        if (len(ids) != count or len(records) != count or ids & seen
                or any(type(r["is_metal"]) is not bool for r in records) or sum(r["is_metal"] for r in records)*2 != count):
            raise ValueError("Frozen balanced cohort invalid")
        seen |= ids
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir()
    config = dict(seeds=args.seeds, doses=sorted(doses), supervised_epochs=args.supervised_epochs, arms=list(ARMS),
                  warmup_batching=batching, train_limits=TRAIN_LIMITS, eval_limits=EVAL_LIMITS,
                  model=MODEL, graph=dict(radius=6., max_num_neighbors=120, device="cpu"),
                  optimizer="AdamW; lr=1e-4; weight_decay=0 warmup / 1e-5 supervised; clip=1; reset at handover",
                  update_scope="full scratch backbone and classifier", evaluation="validation only; no test or temperature scaling",
                  seed_streams=dict(initial="seed", first_bank="100000+seed", later_banks="stable_seed('epoch_bank',seed,epoch)",
                                    warmup="300000+seed", warmup_order="400000+seed+epoch", labels="500000+seed",
                                    supervised="600000+seed", supervised_order="700000+seed+epoch"),
                  device=device, python=platform.python_version(),
                  versions={p: importlib.metadata.version(p) for p in ("torch", "orb-models", "ase", "numpy", "scipy", "pandas", "matplotlib", "nvalchemi-toolkit-ops", "warp-lang")},
                  data_sha256={s: sha256(HERE / "data" / f"{s}.json") for s in ("train", "val")},
                  code_sha256={f: sha256(HERE / f) for f in ("run_methods.py", "run_doses.py", "plot.py")})
    save_json(output / "config.json", config)
    train, val = build_items(clean), build_items(validation)
    warm_rows, sup_rows = [], []
    for seed in args.seeds:
        initial = new_model(seed)
        check_contract(initial, train, device)
        phase(initial, train, val, None, seed, ARMS[0], doses, args.supervised_epochs, device, output, sup_rows)
        banks = Banks(clean, seed, max(doses), output, batching)
        for arm in ARMS[1:]:
            states = phase(initial, train, val, banks, seed, arm, doses, args.supervised_epochs, device, output, warm_rows)
            for dose in sorted(doses):
                start = copy.deepcopy(initial)
                start.load_state_dict(torch.load(states[dose], map_location="cpu", weights_only=True))
                phase(start, train, val, None, seed, arm, doses, args.supervised_epochs, device, output, sup_rows, dose)
                del start
        del initial, banks
        release_cache(device)
    check_pairing(pd.DataFrame(warm_rows), pd.DataFrame(sup_rows))
    plot(output)
    save_json(output / "COMPLETE.json", dict(seeds=args.seeds, doses=doses, test_data_access="NONE"))
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
    run(args, [args.warmup_epochs], batching="all_banks")
