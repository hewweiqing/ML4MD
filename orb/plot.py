"""Four-panel ORB validation dynamics from saved tables; no Torch/ORB imports."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import rankdata, t

ARMS = ("control", "fixed_inputs_fixed_labels", "fixed_inputs_fresh_labels", "fresh_inputs_fresh_labels")
QUANTITIES = ("mean_confidence", "accuracy", "nll", "mean_predictive_entropy", "ece", "auroc")


def metrics(logits, labels, probabilities=None, nll=None):
    logp = logits-logsumexp(logits, axis=1, keepdims=True)
    probs = np.exp(logp) if probabilities is None else probabilities
    p = probs[:, 1]
    confidence, prediction = np.maximum(p, 1-p), logits.argmax(axis=1)
    positive = labels == 1
    npos, nneg = int(positive.sum()), int((~positive).sum())
    if npos == 0 or nneg == 0:
        raise ValueError("Validation must contain both classes")
    auc = (rankdata(p)[positive].sum()-npos*(npos+1)/2)/(npos*nneg)
    ece, edges = 0., np.linspace(0., 1., 16)
    correctness = (p > .5) == labels
    for i in range(15):
        mask = (confidence >= edges[i]) & ((confidence <= edges[i+1]) if i == 14 else (confidence < edges[i+1]))
        if mask.any():
            ece += float(mask.mean())*abs(float(correctness[mask].mean())-float(confidence[mask].mean()))
    result = dict(nll=float(-logp[np.arange(len(labels)), labels].mean()) if nll is None else nll,
                  accuracy=float((prediction == labels).mean()), auroc=float(auc), brier=float(((p-labels)**2).mean()),
                  ece=ece, mean_confidence=float(confidence.mean()),
                  mean_predictive_entropy=float(-(probs*np.log(np.clip(probs, 1e-15, 1.))).sum(1).mean()),
                  mean_absolute_logit_margin=float(np.abs(logits[:, 1]-logits[:, 0]).mean()),
                  predicted_metal_fraction=float((prediction == 1).mean()), records=len(labels))
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Nonfinite validation metrics")
    return result


def summarize(warm, supervised, config, dose):
    rows, seeds = [], config["seeds"]
    for arm in ARMS:
        sup = supervised[(supervised.arm == arm) & (supervised.dose == (0 if arm == "control" else dose))].copy()
        if arm == "control":
            pre = pd.concat([sup[sup.epoch == 0].assign(epoch=e) for e in range(dose+1)])
        else:
            pre = warm[(warm.arm == arm) & (warm.epoch <= dose)].copy()
        pre["protocol_epoch"] = pre.epoch-dose
        sup["protocol_epoch"] = sup.epoch
        # Match original ORB: use warm-up endpoint at handover, then supervised 1..20.
        combined = pd.concat([pre, sup[sup.epoch > 0]])
        expected = {(s, e) for s in seeds for e in range(-dose, config["supervised_epochs"]+1)}
        actual = list(combined[["seed", "protocol_epoch"]].itertuples(index=False, name=None))
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError(f"Incomplete/duplicate trajectories for {arm}, E={dose}; no partial-seed figure produced")
        for epoch, group in combined.groupby("protocol_epoch"):
            for metric in QUANTITIES:
                values = group[metric].to_numpy(float)
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite {metric}")
                half = t.ppf(.975, len(seeds)-1)*values.std(ddof=1)/np.sqrt(len(seeds)) if len(seeds)>1 else 0.
                rows.append(dict(warmup_epochs=dose, policy=arm, protocol_epoch=epoch, metric=metric,
                                 n=len(seeds), mean=values.mean(), low=values.mean()-half, high=values.mean()+half))
    return pd.DataFrame(rows)


def plot(root):
    root = Path(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    warm, supervised = [pd.read_csv(root / f"{s}.csv", float_precision="round_trip") for s in ("warmup", "supervised")]
    # Validate every requested dose before writing any figures.
    summaries = [summarize(warm, supervised, config, dose) for dose in config["doses"]]
    out = root / "figures"
    out.mkdir(exist_ok=True)
    titles = ("Control", "Fixed inputs / fixed labels", "Fixed inputs / fresh labels", "Fresh inputs / fresh labels")
    for dose, summary in zip(config["doses"], summaries):
        for kind, curves in (("confidence_accuracy", (("mean_confidence", "Mean confidence", "tab:blue"), ("accuracy", "Accuracy", "tab:orange"))),
                             ("nll", (("nll", "NLL", "tab:blue"),))):
            fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True, constrained_layout=True)
            for ax, arm, title, letter in zip(axes, ARMS, titles, "abcd"):
                for metric, label, color in curves:
                    data = summary[(summary.policy == arm) & (summary.metric == metric)]
                    ax.plot(data.protocol_epoch, data["mean"], color=color, lw=1.7, label=label)
                    ax.fill_between(data.protocol_epoch, data.low, data.high, color=color, alpha=.15, linewidth=0)
                ax.axvspan(-dose, 0, color="0.92" if arm == "control" else "#f0e8fa", zorder=-1)
                ax.axvline(0, color=".5", ls="--", lw=1)
                ax.set(title=title, xlabel="Warm-up passes | supervised epoch", xlim=(-dose, config["supervised_epochs"]))
                ticks = sorted(set(range(5, config["supervised_epochs"]+1, 5)) | {config["supervised_epochs"]})
                ax.set_xticks([-dose, -dose/2, 0]+ticks, ["0", f"{dose/2:g}", f"{dose} | 0"]+[str(x) for x in ticks])
                ax.spines[["top", "right"]].set_visible(False)
                ax.text(-.08, 1.06, letter, transform=ax.transAxes, weight="bold")
                if kind == "confidence_accuracy":
                    ax.set_ylim(0, 1.02)
            axes[0].set_ylabel("Balanced-validation metric" if kind == "confidence_accuracy" else "Validation NLL")
            axes[0].legend(frameon=False, fontsize=8)
            for extension in ("png", "pdf"):
                fig.savefig(out / f"ORB_W{dose:03d}_{kind}.{extension}", dpi=250)
            plt.close(fig)
    pd.concat(summaries).to_csv(out / "DYNAMICS_SUMMARY.csv", index=False)
    (out / "caption.txt").write_text(
        "Real balanced-validation materials (252), not synthetic-bank evaluation or test. "
        "Means and pointwise 95% Student-t intervals across paired seeds. The control's warm-up "
        "segment repeats its initialization evaluation; no control warm-up updates occur. "
        "Other arms show measured epochs. At handover the warm-up endpoint is used once, "
        "followed by supervised epochs 1 onward. No checkpoint selection or temperature scaling. "
        "Input/label policies are distinct from ALIGNN and CrabNet; graph-bounded updates per "
        "pass need not match across architectures. Exploratory validation-only study.\n", encoding="utf-8")
    print(f"Figures: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    plot(p.parse_args().root)
