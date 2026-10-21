"""Shared metrics and four-panel plots."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import t
from sklearn.metrics import roc_auc_score


def metrics(logits, labels):
    z, y = np.asarray(logits, dtype=float), np.asarray(labels, dtype=float)
    p = expit(z)
    return dict(nll=float(np.mean(np.logaddexp(0, z)-y*z)),
                accuracy=float(np.mean((p>=.5)==y)),
                mean_confidence=float(np.maximum(p, 1-p).mean()),
                mean_abs_logit=float(np.abs(z).mean()),
                auroc=float(roc_auc_score(y, z)))


def summarize(frame, seeds):
    rows = []
    for time, group in frame.groupby("time", sort=True):
        if sorted(group.seed.tolist()) != seeds:
            raise ValueError("Missing or duplicate seeds in a plotted checkpoint")
        row = {"time": int(time)}
        for name in ("mean_confidence", "accuracy"):
            values = group[name].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite figure data")
            half = t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)) if len(values)>1 else 0.
            row.update({name: values.mean(), name+"_low": values.mean()-half, name+"_high": values.mean()+half})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_results(folder, output=None):
    folder = Path(folder)
    output = Path(output) if output else folder/"figures"
    config = json.loads((folder/"config.json").read_text(encoding="utf-8"))
    seeds, doses = config["seeds"], config["doses"]
    warm = pd.read_csv(folder/"warmup.csv")
    supervised = pd.read_csv(folder/"supervised.csv")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
        "figure.constrained_layout.use": True})
    panels = [("control", "Control"), ("U_fixed", "Unique-element bank"),
              ("A_fixed", "Atom-count bank"), ("U_refreshed", "Refreshed unique-element bank")]
    for dose in doses:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
        all_summaries = []
        for letter, ax, (arm, title) in zip("abcd", axes, panels):
            if arm=="control":
                start = warm[(warm.arm=="U_fixed") & (warm.passes==0)].copy()
                start["time"] = 0
            else:
                start = warm[(warm.arm==arm) & (warm.passes<=dose)].copy()
                start["time"] = start.passes-dose
            branch = supervised[(supervised.arm==arm) & (supervised.dose==(0 if arm=="control" else dose))].copy()
            if set(branch.epoch) != set(range(1, config["supervised_epochs"]+1)):
                raise ValueError("Incomplete supervised trajectory")
            branch["time"] = branch.epoch
            summary = summarize(pd.concat([start, branch]), seeds)
            summary["arm"] = arm
            all_summaries.append(summary)
            ax.axvspan(-dose, 0, color="0.93" if arm=="control" else "#eadcf8",
                       alpha=1 if arm=="control" else .55, zorder=0)
            for name, color, label, marker in [("mean_confidence", "C0", "Mean confidence", "o"),
                                              ("accuracy", "C1", "Accuracy", "s")]:
                ax.plot(summary.time, summary[name], color=color, marker=marker, ms=3, lw=1.6, label=label)
                ax.fill_between(summary.time.to_numpy(), summary[name+"_low"].to_numpy(),
                                summary[name+"_high"].to_numpy(), color=color, alpha=.15, linewidth=0)
            ax.axvline(0, color="0.35", ls="--", lw=1)
            ax.set_title(title if arm=="control" else f"{title}, E={dose}")
            ax.set_xlim(-dose-.5, config["supervised_epochs"]+.5)
            ax.set_ylim(.45, 1)
            ticks = list(range(5, config["supervised_epochs"]+1, 5))
            ax.set_xticks([-dose, dose//2-dose, 0]+ticks)
            ax.set_xticklabels(["0", str(dose//2), f"{dose} | 0"]+[str(v) for v in ticks])
            ax.set_xlabel("Warm-up passes  |  supervised epoch")
            ax.text(-.13, 1.05, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
        axes[0].set_ylabel("Real balanced-validation metric")
        axes[0].legend(frameon=False, loc="upper left", fontsize=8)
        name = output/f"confidence_accuracy_E{dose:02d}"
        for suffix in (".png", ".pdf"):
            fig.savefig(name.with_suffix(suffix), dpi=300, bbox_inches="tight")
        plt.close(fig)
        pd.concat(all_summaries).to_csv(output/f"curve_summary_E{dose:02d}.csv", index=False)
        print(f"Saved {name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="An experiment folder containing warmup.csv and supervised.csv")
    parser.add_argument("--output", type=Path, help="Optional separate figure destination")
    args = parser.parse_args()
    plot_results(args.folder, args.output)
