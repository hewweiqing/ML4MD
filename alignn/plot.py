"""Metrics and figures; plot saved CSVs without importing PyTorch or DGL."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def metrics(logp, labels):
    prob = np.exp(logp[:, 1])
    return dict(auroc=float(roc_auc_score(labels, prob)), loss=float(log_loss(labels, prob, labels=[0, 1])),
                accuracy=float(accuracy_score(labels, prob > 0.5)), pred_std=float(prob.std()),
                nll=float(-logp[np.arange(len(labels)), labels].mean()),
                mean_confidence=float(np.maximum(prob, 1-prob).mean()), brier=float(((prob-labels)**2).mean()))


def aggregate(frame, time, quantities, seeds):
    rows = []
    for x, group in frame.groupby(time):
        if len(group) != len(seeds) or set(group.seed) != set(seeds):
            raise ValueError(f"Missing or duplicate seeds at {time}={x}")
        row = {time: x}
        for quantity in quantities:
            values = group[quantity].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite {quantity}")
            half = t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)) if len(values)>1 else 0.
            row.update({quantity: values.mean(), quantity+"_low": values.mean()-half,
                        quantity+"_high": values.mean()+half})
        rows.append(row)
    return pd.DataFrame(rows)


def trace(ax, data, x, quantity, label, color, linestyle="-"):
    ax.plot(data[x], data[quantity], color=color, ls=linestyle, lw=1.8, label=label)
    ax.fill_between(data[x].to_numpy(), data[quantity+"_low"].to_numpy(),
                    data[quantity+"_high"].to_numpy(), color=color, alpha=.15, linewidth=0)


def plot(root):
    root = Path(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    seeds = config["seeds"]
    warm, supervised = (pd.read_csv(root / f"{name}.csv", float_precision="round_trip") for name in ("warmup", "supervised"))
    branches = [("control", 0)] + [(arm, dose) for arm in ("F", "R") for dose in config["doses"]]
    expected = {(seed, arm, dose, epoch) for seed in seeds for arm, dose in branches
                for epoch in range(config["supervised_epochs"]+1)}
    actual = list(supervised[["seed", "arm", "dose", "epoch"]].itertuples(index=False, name=None))
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete or duplicate supervised trajectories; wait for training to finish")
    out = root / "figures"
    out.mkdir(exist_ok=True)
    titles = ("Control", "Fixed labels (F)", "Refreshed labels (R)")
    for dose in config["doses"]:
        summaries = []
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True, constrained_layout=True)
        for ax, arm, title, letter in zip(axes, ("control", "F", "R"), titles, "abc"):
            sup = supervised[(supervised.arm == arm) & (supervised.dose == (0 if arm == "control" else dose))].copy()
            sup["protocol_epoch"] = sup.epoch
            # Supervised epoch 0 is the handover; do not count it twice.
            pre = warm[(warm.arm == arm) & (warm.epoch < dose)].copy()
            pre["protocol_epoch"] = pre.epoch-dose
            data = aggregate(pd.concat([pre, sup]), "protocol_epoch", ("mean_confidence", "accuracy"), seeds)
            trace(ax, data, "protocol_epoch", "mean_confidence", "Mean confidence", "tab:blue")
            trace(ax, data, "protocol_epoch", "accuracy", "Accuracy", "tab:orange")
            ax.axvspan(-dose, 0, color="0.92" if arm == "control" else "#f0e8fa", zorder=-1)
            ax.axvline(0, color="0.5", ls="--", lw=1)
            ax.set(title=title, xlabel="Warm-up passes | supervised epoch", ylim=(0, 1.02),
                   xlim=(-dose, config["supervised_epochs"]))
            ticks = sorted(set(range(5, config["supervised_epochs"]+1, 5)) | {config["supervised_epochs"]})
            ax.set_xticks([-dose, -dose/2, 0] + ticks,
                          ["0", f"{dose/2:g}", f"{dose} | 0"] + [str(tick) for tick in ticks])
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(-.10, 1.05, letter, transform=ax.transAxes, weight="bold")
            summaries.append(data.assign(arm=arm, dose=dose))
        axes[0].set_ylabel("Balanced-validation metric")
        axes[0].legend(frameon=False, fontsize=8, loc="lower right")
        for extension in ("png", "pdf"):
            fig.savefig(out / f"confidence_accuracy_E{dose:02d}.{extension}", dpi=250)
        plt.close(fig)
        pd.concat(summaries).to_csv(out / f"confidence_accuracy_E{dose:02d}_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    summaries = []
    for ax, arm in zip(axes, ("F", "R")):
        data = aggregate(warm[warm.arm == arm], "optimizer_updates", ("on_bank_nll", "off_bank_nll", "memorization_gap"), seeds)
        trace(ax, data, "optimizer_updates", "on_bank_nll", "Training bank", "tab:blue")
        trace(ax, data, "optimizer_updates", "off_bank_nll", "Independent bank", "tab:orange", "--")
        ax.axhline(np.log(2), color="0.5", ls=":", label="ln 2")
        ax.set(title=f"Arm {arm}", xlabel="Warm-up optimizer updates", ylabel="NLL")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        summaries.append(data.assign(arm=arm))
    for extension in ("png", "pdf"):
        fig.savefig(out / f"synthetic_bank_nll.{extension}", dpi=250)
    plt.close(fig)
    pd.concat(summaries).to_csv(out / "synthetic_bank_nll_summary.csv", index=False)
    (out / "caption.txt").write_text(
        "Confidence and accuracy: real balanced validation set (252 materials). "
        "Lines are seed means; bands are pointwise 95% Student-t intervals, not pooled samples. "
        "Control has no warm-up (blank shaded region). F fixes structures and random labels; "
        "R fixes the same structures but redraws labels each update. R bank NLL is the exact "
        "expected NLL over independent Bernoulli(0.5) labels, not realized training-batch loss. "
        "F bank NLL uses fixed bank-specific labels. Memorization gaps are calculated within seed. "
        "No temperature scaling. Test metrics are separate in test.csv. "
        "Exploratory balanced-cohort sensitivity; test materials had been inspected historically.\n",
        encoding="utf-8")
    print(f"Figures: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    plot(p.parse_args().root)
