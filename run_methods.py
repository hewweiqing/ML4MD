"""Compare Control, fixed U, fixed A and refreshed U at one warm-up dose."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
from pathlib import Path
import random
import types

import numpy as np
import pandas as pd
from pymatgen.core import Composition
import torch
from torch.utils.data import DataLoader, TensorDataset
from crabnet import kingcrab
from crabnet.crabnet_ import CrabNet
from crabnet.utils.optim import SWA
from crabnet.utils.utils import BCEWithLogitsLoss, DummyScaler, EDM_CsvLoader, Lamb, Lookahead

from plot import metrics, plot_results

ROOT = Path(__file__).resolve().parent
ARMS = ("U_fixed", "A_fixed", "U_refreshed")
BATCH_SIZE = 128
SUPERVISED_EPOCHS = 20


# CrabNet 2.0.8's optimizer wrappers need these attributes under PyTorch 2.x.
def patch_optimizer(cls):
    if getattr(cls, "_torch2_compat_patched", False):
        return
    original = cls.__init__
    @wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        for name in ("_optimizer_step_pre_hooks", "_optimizer_step_post_hooks"):
            if not hasattr(self, name):
                setattr(self, name, OrderedDict())
        self._zero_grad_profile_name = f"Optimizer.zero_grad#{type(self).__name__}.zero_grad"
    cls.__init__ = initialize
    cls._torch2_compat_patched = True


patch_optimizer(SWA)
patch_optimizer(Lookahead)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bank_seed(seed, role, stream, refreshed=False):
    version = "crabnet-random2-direct-v1" if refreshed else "crabnet-fixed-bank-v1"
    return int.from_bytes(hashlib.sha256(f"{version}|{seed}|{role}|{stream}".encode()).digest()[:8], "little")


@contextmanager
def preserve_rng():
    """Diagnostics must not change subsequent dropout or data-order randomness."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def load_split(name):
    rows = json.loads((ROOT/"data"/f"{name}.json").read_text(encoding="utf-8"))
    return pd.DataFrame({"sample_id": [r["mbid"] for r in rows],
                         "formula": [r["composition"].replace(" ", "") for r in rows],
                         "target": [float(r["is_metal"]) for r in rows]})


def template_counts(train):
    unique, atoms = [], []
    for formula in train.formula:
        composition = Composition(formula)
        amounts = np.asarray(list(composition.reduced_composition.get_el_amt_dict().values()))
        if (amounts < 1).any() or not np.allclose(amounts, np.rint(amounts), atol=1e-8, rtol=0):
            raise ValueError(f"Non-integer reduced stoichiometry: {formula}")
        unique.append(len(composition.elements))
        atoms.append(int(np.rint(amounts).sum()))
    print(f"Reduced atom counts: median={np.median(atoms):g}, p95={np.percentile(atoms, 95):g}, max={max(atoms)}")
    if max(atoms) > 150:
        raise ValueError("Atom counts exceed the audited cohort maximum (150); inspect them before running. No capping.")
    return unique, atoms


def make_bank(counts, vocabulary_size, width, seed, arm, role="fixed", epoch=1):
    """Use the original independent RNG streams, including shared fixed U/A labels."""
    refreshed = arm == "U_refreshed"
    if refreshed:
        element_seed = bank_seed(10_000+seed, epoch, "elements", True)
        fraction_seed = bank_seed(10_000+seed, epoch, "fractions", True)
        label_seed = bank_seed(10_000+seed, epoch, "labels", True)
    else:
        element_seed = bank_seed(seed, role, "atoms_with_replacement" if arm=="A_fixed" else "unique_elements")
        fraction_seed = bank_seed(seed, role, "unique_fractions")
        label_seed = bank_seed(seed, "fresh_shared" if role=="fresh" else "fixed_shared", "shared_labels")
    elements_rng = np.random.default_rng(element_seed)
    fractions_rng = np.random.default_rng(fraction_seed)
    rows = []
    for count in counts:
        if arm == "A_fixed":
            draws = elements_rng.integers(0, vocabulary_size, size=count)
            elements, first, multiplicity = np.unique(draws, return_index=True, return_counts=True)
            order = np.argsort(first)
            rows.append((elements[order]+1, (multiplicity[order]/count).astype(np.float32)))
        else:
            elements = elements_rng.choice(vocabulary_size, size=count, replace=False)+1
            fractions = fractions_rng.dirichlet(np.ones(count)).astype(np.float32)
            fractions /= fractions.sum(dtype=np.float64)
            rows.append((elements, fractions))
    width = max(len(row[0]) for row in rows) if arm=="A_fixed" else width
    src = np.zeros((len(rows), width), dtype=np.int64)
    frac = np.zeros((len(rows), width), dtype=np.float32)
    for i, (elements, fractions) in enumerate(rows):
        src[i, :len(elements)], frac[i, :len(fractions)] = elements, fractions
    labels = np.random.default_rng(label_seed).binomial(1, .5, len(rows)).astype(np.float32)
    return TensorDataset(torch.from_numpy(src), torch.from_numpy(frac), torch.from_numpy(labels))


def check_bce(model):
    if model.classification is not True or model.criterion is not BCEWithLogitsLoss or not isinstance(model.scaler, DummyScaler):
        raise RuntimeError("Expected binary classification, BCEWithLogitsLoss and unscaled targets")


def new_model(seed, train, val, device, epochs=SUPERVISED_EPOCHS):
    set_seed(seed)
    model = CrabNet(classification=True, criterion="BCEWithLogitsLoss", epochs=0,
                    epochs_step=10, batch_size=BATCH_SIZE, checkin=1, losscurve=False,
                    learningcurve=False, save=False, random_state=seed, verbose=False,
                    compute_device=device)
    # CrabNet 2.0.8 overwrites the constructor flag. Restore it BEFORE first fit.
    model.classification = True
    model.fit(train_df=train, val_df=val)
    check_bce(model)
    if model.model.encoder.embed.cbfv.weight.requires_grad:
        raise RuntimeError("Native element embedding must remain frozen")
    model.optimizer = model.lr_scheduler = None
    model.epochs = epochs
    return model


def snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.model.state_dict().items()}


def predict(model, frame):
    loader = EDM_CsvLoader(frame[["formula", "target"]].copy(), extra_features=None,
        batch_size=BATCH_SIZE, n_elements=model.n_elements, inference=True,
        verbose=False, elem_prop=model.elem_prop).get_data_loaders(inference=True)
    was_training = model.model.training
    model.model.eval()
    logits, labels = [], []
    try:
        with torch.no_grad():
            for x, y, _, extra in loader:
                src, frac = x.squeeze(-1).chunk(2, dim=1)
                output = model.model(src.to(model.compute_device, dtype=torch.long),
                    frac.to(model.compute_device, dtype=model.data_type_torch),
                    extra_features=extra.to(model.compute_device, dtype=model.data_type_torch))
                logits.extend(output[:, 0].cpu().tolist())
                labels.extend(y.view(-1).tolist())
    finally:
        model.model.train(was_training)
    return np.asarray(logits), np.asarray(labels)


def bank_nll(model, bank):
    width = bank.tensors[0].shape[1]
    loader = DataLoader(bank, batch_size=min(BATCH_SIZE, max(8, BATCH_SIZE*16//width)), shuffle=False)
    was_training = model.model.training
    model.model.eval()
    logits, labels = [], []
    try:
        with torch.no_grad():
            for src, frac, y in loader:
                z = model.model(src.to(model.compute_device), frac.to(model.compute_device),
                    extra_features=torch.zeros((len(y), 0), device=model.compute_device))[:, 0]
                logits.append(z.cpu().double())
                labels.append(y.double())
    finally:
        model.model.train(was_training)
    return torch.nn.functional.binary_cross_entropy_with_logits(torch.cat(logits), torch.cat(labels)).item()


def warmup(model, val, bank, independent, seed, arm, doses, refreshed_bank):
    """One uninterrupted trajectory; all requested doses are exact prefixes."""
    optimizer = Lookahead(Lamb(model.model.parameters(), lr=model.lr, betas=model.betas,
        eps=model.eps, weight_decay=model.weight_decay, adam=model.adam,
        min_trust=model.min_trust), alpha=model.alpha, k=model.k)
    set_seed(2_000_000+seed)
    rows, checkpoints = [], {}
    for epoch in range(max(doses)+1):
        if epoch:
            check_bce(model)
            if arm == "U_refreshed":
                bank = refreshed_bank(epoch)
                order_seed = bank_seed(10_000+seed, epoch, "batch_order", True)
            else:
                order_seed = bank_seed(seed, "fixed", "shared_batch_order")
            loader = DataLoader(bank, batch_size=BATCH_SIZE, shuffle=True,
                                generator=torch.Generator().manual_seed(order_seed))
            model.model.train()
            microbatch = BATCH_SIZE if arm=="U_refreshed" else 16
            for src, frac, labels in loader:
                if arm == "U_refreshed":
                    frac = model._add_jitter(src, frac)
                for start in range(0, len(labels), microbatch):
                    x = src[start:start+microbatch].to(model.compute_device)
                    f = frac[start:start+microbatch].to(model.compute_device)
                    y = labels[start:start+microbatch].to(model.compute_device)
                    pred, uncertainty = model.model(x, f, extra_features=torch.zeros(
                        (len(y), 0), device=model.compute_device)).chunk(2, dim=-1)
                    loss = model.criterion(pred.view(-1), uncertainty.view(-1), y)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite warm-up loss")
                    (loss*(len(y)/len(labels))).backward()
                optimizer.step()
                optimizer.zero_grad()
            if epoch in doses:
                checkpoints[epoch] = snapshot(model)
        with preserve_rng():
            row = dict(seed=seed, arm=arm, passes=epoch,
                       optimizer_updates=epoch*((len(bank)+BATCH_SIZE-1)//BATCH_SIZE),
                       **metrics(*predict(model, val)))
            row["on_bank_nll"] = bank_nll(model, bank)
            row["off_bank_nll"] = bank_nll(model, independent)
            row["memorization_gap"] = row["off_bank_nll"]-row["on_bank_nll"]
        rows.append(row)
        print(f"seed {seed}: {arm} warm-up {epoch}/{max(doses)}", flush=True)
    return checkpoints, rows


def supervised(model, train, val, seed, arm, dose):
    """Native optimizer/scheduler/SWA; explicitly restore train mode each epoch."""
    rows = []
    load_original = model._load_trainval_data
    def load(self, *args):
        load_original(*args)
        self.train_loader.sampler.generator = torch.Generator().manual_seed(1_000_000+seed)
    def train_epoch(self):
        check_bce(self)
        self.model.train()
        for x, y, _, extra in self.train_loader:
            src, frac = x.squeeze(-1).chunk(2, dim=1)
            frac = self._add_jitter(src, frac)
            pred, uncertainty = self.model(src.to(self.compute_device, dtype=torch.long),
                frac.to(self.compute_device), extra_features=extra.to(self.compute_device)).chunk(2, dim=-1)
            loss = self.criterion(pred.view(-1), uncertainty.view(-1), self.scaler.scale(y).to(self.compute_device).view(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite supervised loss")
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.stepping:
                self.lr_scheduler.step()
        if (self.epoch+1) % (2*self.epochs_step)==0 and self.epoch >= self.epochs_step*self.swa_start-1:
            from scipy.special import expit
            logits, labels = predict(self, val)
            self.optimizer.update_swa(float(np.mean(np.abs(expit(logits)-labels))))
    def stats(self, epochs, epoch):
        row = dict(seed=seed, arm=arm, dose=dose, epoch=epoch+1, **metrics(*predict(self, val)))
        rows.append(row)
        self.loss_curve["train"].append(float("nan"))
        self.loss_curve["val"].append(1-row["accuracy"])
        self.model.train()
        print(f"seed {seed}: {arm} E={dose}, supervised {epoch+1}/{epochs}", flush=True)
    model._load_trainval_data = types.MethodType(load, model)
    model._train = types.MethodType(train_epoch, model)
    model._losscurve_stats = types.MethodType(stats, model)
    model._track_stats = types.MethodType(lambda self, *args: None, model)
    set_seed(1_000_000+seed)
    model.classification = True
    model.fit(train_df=train, val_df=val)
    check_bce(model)
    return rows


def run(doses, seeds, output, device):
    import gc
    import importlib.metadata
    import platform
    doses, seeds, output = sorted(set(doses)), sorted(set(seeds)), Path(output).resolve()
    if not doses or min(doses)<1 or not seeds or min(seeds)<0:
        raise ValueError("Use positive doses and nonnegative seeds")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output is not empty. Choose a new --output folder; existing results are never overwritten.")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device=="auto" else device
    if device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    train, val = load_split("train"), load_split("val")
    unique, atoms = template_counts(train)
    for frame in (train, val):
        if not frame.sample_id.is_unique or frame.target.mean()!=.5:
            raise ValueError("Expected unique IDs and balanced labels")
    if set(train.sample_id) & set(val.sample_id):
        raise ValueError("Train/validation overlap")
    embedding_path = Path(kingcrab.__file__).parent/"data/element_properties/mat2vec.csv"
    vocabulary = pd.read_csv(embedding_path, index_col=0)
    output.mkdir(parents=True, exist_ok=True)
    (output/"checkpoints").mkdir()
    config = dict(doses=doses, seeds=seeds, arms=list(ARMS), device=device, batch_size=BATCH_SIZE,
        supervised_epochs=SUPERVISED_EPOCHS, deterministic_algorithms=True,
        controls="trained once per seed; reused across doses", temperature_scaling=False,
        python_version=platform.python_version(),
        embedding_sha256=hashlib.sha256(embedding_path.read_bytes()).hexdigest(),
        data_sha256={name: hashlib.sha256((ROOT/"data"/f"{name}.json").read_bytes()).hexdigest()
                     for name in ("train", "val", "test")},
        versions={name: importlib.metadata.version(name) for name in
                  ("crabnet", "torch", "numpy", "pandas", "pymatgen", "scipy", "scikit-learn", "matplotlib")})
    (output/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    warm_rows, supervised_rows = [], []
    print(f"Device: {device}. Seeds: {seeds}. Warm-up passes: {doses}.", flush=True)
    for seed in seeds:
        model = new_model(seed, train, val, device)
        initial = snapshot(model)
        width = int(model.n_elements)
        banks = {(arm, role): make_bank(atoms if arm=="A_fixed" else unique, len(vocabulary), width, seed, arm, role)
                 for arm in ("U_fixed", "A_fixed") for role in ("fixed", "fresh")}
        for role in ("fixed", "fresh"):
            if not torch.equal(banks["U_fixed", role].tensors[2], banks["A_fixed", role].tensors[2]):
                raise RuntimeError("U/A labels differ")
        supervised_rows.extend(supervised(model, train, val, seed, "control", 0))
        torch.save(snapshot(model), output/"checkpoints"/f"seed_{seed:02d}_control_E0.pt")
        del model
        for arm in ARMS:
            model = new_model(seed, train, val, device)
            model.model.load_state_dict(initial)
            def refreshed(epoch):
                return make_bank(unique, len(vocabulary), width, seed, "U_refreshed", epoch=epoch)
            bank = refreshed(1) if arm=="U_refreshed" else banks[arm, "fixed"]
            independent = banks["U_fixed" if arm=="U_refreshed" else arm, "fresh"]
            checkpoints, rows = warmup(model, val, bank, independent, seed, arm, doses, refreshed)
            warm_rows.extend(rows)
            pd.DataFrame(warm_rows).to_csv(output/"warmup.csv", index=False)
            del model
            for dose in doses:
                model = new_model(seed, train, val, device)
                model.model.load_state_dict(checkpoints[dose])
                supervised_rows.extend(supervised(model, train, val, seed, arm, dose))
                torch.save(snapshot(model), output/"checkpoints"/f"seed_{seed:02d}_{arm}_E{dose}.pt")
                pd.DataFrame(supervised_rows).to_csv(output/"supervised.csv", index=False)
                del model
            del checkpoints
            gc.collect()
        del initial, banks
    # No test inference until ALL seeds, doses and branches have finished training.
    test = load_split("test")
    if test.target.mean()!=.5 or not test.sample_id.is_unique or set(test.sample_id)&set(pd.concat([train, val]).sample_id):
        raise ValueError("Unbalanced or overlapping test split")
    test_rows = []
    for seed in seeds:
        model = new_model(seed, train, val, device)
        for arm, dose in [("control", 0)]+[(a, d) for a in ARMS for d in doses]:
            model.model.load_state_dict(torch.load(output/"checkpoints"/f"seed_{seed:02d}_{arm}_E{dose}.pt", map_location=device))
            test_rows.append(dict(seed=seed, arm=arm, dose=dose, **metrics(*predict(model, test))))
        del model
    pd.DataFrame(test_rows).to_csv(output/"test.csv", index=False)
    plot_results(output)
    print(f"Finished: {output}", flush=True)


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser


if __name__ == "__main__":
    parser = arguments(__doc__)
    parser.add_argument("--warmup-epochs", type=int, default=16)
    parser.add_argument("--output", type=Path, default=ROOT/"runs/methods")
    args = parser.parse_args()
    run([args.warmup_epochs], args.seeds, args.output, args.device)
