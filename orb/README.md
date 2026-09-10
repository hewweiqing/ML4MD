# ORB warm-up experiments

Minimal runners for the **balanced-cohort methods and 5/15/30-pass studies**.
They do not depend on any old experiment folder, trained checkpoint, or `results/`.
The older unbalanced, descriptor-space and 110-epoch convergence studies are not
part of these two runners.

| File | Purpose |
| --- | --- |
| `run_methods.py` | Four methods at 4 warm-up passes; shared model, noise and training code. |
| `run_doses.py` | The same methods at 5, 15 and 30 passes; imports shared code. |
| `plot.py` | Four-panel confidence/accuracy and NLL figures from saved CSVs; no Torch/ORB imports. |

## Install

Use **Python 3.12** in a separate environment, not the Python 3.10 ALIGNN environment.
From this folder:

```sh
python -m pip install -r requirements.txt
```

The original ORB runtime used PyTorch 2.8.0 with CUDA 12.8. For that NVIDIA build:

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

See [PyTorch's version-specific installation commands](https://pytorch.org/get-started/previous-versions/)
for CPU or other supported builds. `environment.txt` records the original
environment's `pip list --format=freeze`, without local build paths. It contains
unrelated packages: install from `requirements.txt`, not `environment.txt`.

No Windows paths are embedded in the runners. Compiled dependencies still have
platform requirements: [ORB's upstream documentation](https://github.com/orbital-materials/orb-models)
does not guarantee Windows support, although this experiment's original runtime
is Windows. This package does not claim a clean install was tested on every OS.
The pinned graph backend matters for neighbour ordering and batch composition.
No pretrained weights are downloaded: ORB-v3 is initialized **from scratch**.

## Run

```sh
python -u run_methods.py
python -u run_doses.py
```

Run sequentially. CUDA is selected when available, otherwise CPU; use
`--device cuda` or `--device cpu` to require one. Five seeds **0-4**, twenty
supervised epochs per branch. Example:

```sh
python -u run_doses.py --device cuda --seeds 0 1 --output runs/example
```

Outputs go to `runs/methods/` or `runs/doses/`. Existing nonempty directories are
refused, not overwritten. These minimal runners **do not resume interrupted runs**
or control background jobs. The old experiment's queue/resume system is not included.
Progress prints each epoch; CSVs are updated as training proceeds.
New runs save `config.json`, `warmup.csv`, `supervised.csv`, noisy `banks/`, milestone
and final `checkpoints/`, and `figures/`. Allow several GB for banks/checkpoints and
RAM for graph caching. GPU memory needs depend on structures; limits are retained,
but a run on another GPU is not guaranteed to fit.

## Experiment

Four arms, all beginning from the same initialization within each seed:

1. **Control:** no warm-up.
2. **Fixed inputs / fixed labels:** reuse the first noisy bank and its random labels.
3. **Fixed inputs / fresh labels:** reuse that bank, redraw labels per minibatch visit.
4. **Fresh inputs / fresh labels:** independent new bank each epoch, labels refreshed
   with the **same label stream and batch boundaries as arm 3**.

Data are the unchanged balanced `mp_is_metal` **2,014 train and 252 validation**
records, exactly half metallic in each, matched to the CrabNet/ALIGNN IDs and order.
**No test data, temperature scaling, early stopping or best-checkpoint selection.**
Validation only measures trajectories. There is intentionally no `data/test.json`.

Synthetic banks retain lattice/site count, draw elements 1-100 and wrapped Gaussian
fractional coordinates N(0.5, 0.15^2). The original 200-attempt 0.5-Angstrom placement
rule, ASE true-periodic-distance audit and deterministic geometry-repair seeds are
retained. Failed structures are regenerated, never accepted with a relaxed floor.
Every refreshed bank starts from clean training structures, not the previous bank.
Random labels are Bernoulli(0.5), not forced to exact class balance.

The **entire scratch ORB-v3 backbone plus Linear(256,2)** is trained using mean
node pooling, raw logits and CrossEntropyLoss. AdamW: lr 1e-4; warm-up weight decay
0; supervised weight decay 1e-5; gradient clipping 1.0. No scheduler. Supervision
resets the optimizer and uses paired true-label sample orders/RNG across branches.
Graph radius is 6 Angstrom with at most 120 neighbours per atom.

The dose sweep uses one continuous 30-pass trajectory per seed/policy, snapshots
at 5/15/30, then fresh twenty-epoch supervised branches. No optimizer/label RNG reset
at warm-up milestones. One control per seed is shared across doses: **50 supervised
branches** total; the 4-pass methods experiment has **20**.

Batching is an important retained difference between the two original studies:

- Training limits: 8 structures, 128 atoms, 24,000 edges. Atom-oversized singletons
  are allowed; the edge limit remains hard. Evaluation: 16 / 256 / 30,000.
- Methods: per-record maximum edges across **all four epoch banks**, shared by arms.
- Doses: per-record maximum across **fixed and current epoch banks**, shared by arms.
  Only those two graph banks are cached; geometry is generated/saved on first use.

Do not equate a pass to a fixed 252 updates: graph-budget batches vary. Per-epoch
`updates` and `cumulative_updates` are saved, along with order/label hashes checked
for within-seed pairing. Both protocols release unused CUDA cache above 3.5 GiB.

## Plot

Figures are made automatically after training, or rerun:

```sh
python plot.py runs/methods
python plot.py runs/doses
```

Files: `figures/ORB_W004_confidence_accuracy.png` for methods, and corresponding
`W005`, `W015`, `W030` figures for doses, plus PDF copies and separate NLL figures.
Four clean panels show real **validation** confidence/accuracy, warm-up to
supervision, with seed means and pointwise 95% Student-t intervals. The control's
pre-handover line repeats initialization (it receives no warm-up). Incomplete seed
groups are rejected rather than plotted as complete results. `DYNAMICS_SUMMARY.csv`
contains the underlying statistics. Plotting needs NumPy/SciPy/pandas/matplotlib only.

No completed reference `results/` is included: at packaging, the original four-pass
study and full five-seed dose sweep did not have completion markers. Partial tables
are not labelled final findings. This does not affect standalone training or plotting
of newly completed runs. The original running/paused experiments are untouched.

## Reproducibility

Python, NumPy and Torch are seeded. Stable SHA-256 seeds control epoch-bank generation
and geometry repair; a dedicated CPU Torch generator supplies fresh labels.
Model initialization, minibatch order, labels, full warm-up snapshots and supervised
RNG resets retain the original protocol. Configuration files record package versions,
source/data hashes, settings and seed streams. The original ORB deterministic cuDNN
settings are retained; native GPU graph operations can still be nondeterministic.
No cross-device bitwise checkpoint equivalence is claimed.

Preparation checks verified the frozen cohorts, an entire archived synthetic bank,
bank prefixes at epochs 1/5/15/30 across all five seeds, geometry-repair rules,
paired batch boundaries, small native CPU training steps, archived validation
metrics and plotting/partial-result rejection. These are not a new full training run
or a guarantee of identical results on other devices.

These are simplified protocol-reproduction scripts, not replacements for the
original evidence or legacy recovery tooling. Treat the studies as exploratory
balanced-cohort sensitivity analyses. Before public upload, choose a code license
and verify Matbench/Materials Project attribution and data redistribution terms.
