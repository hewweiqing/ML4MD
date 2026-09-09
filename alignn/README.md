# ALIGNN warm-up experiments

The two completed **balanced-cohort** studies, packaged for straightforward reruns.
The older exploratory `alignn_scratch_v2.py` update-ladder/head-reset studies are
not included in these two runners.

| File | Purpose |
| --- | --- |
| `run_methods.py` | Control, fixed labels (F), refreshed labels (R), with 4 warm-up passes. Contains the shared training functions. |
| `run_doses.py` | Those same methods at 5, 15 and 30 passes; imports the shared functions. |
| `plot.py` | Three-panel confidence/accuracy figures and two-panel synthetic-bank NLL. No PyTorch or DGL required for plotting. |

## Install

Use **Python 3.10** in a fresh environment, then from this folder:

```sh
python -m pip install -r requirements.txt
```

These are the observed original package versions. `environment.txt` records the
original environment with `pip list --format=freeze`, without local Conda build
paths. It is a reference, not an install lockfile; use `requirements.txt` instead.
Matbench is not needed at runtime because the curated structures are included.

For the original NVIDIA CUDA 11.8 setup, install **both** GPU builds:

```sh
python -m pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install dgl==1.1.1+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html
```

DGL's [official wheel index](https://data.dgl.ai/wheels/cu118/repo.html) supplies
platform-specific wheels selected by pip; no laptop paths appear in the code.
The code is OS-independent, but compiled dependencies still require a compatible
OS/CPU/Python combination. Do not upgrade DGL to 2.x for these scripts. On machines
without supported CUDA, use CPU; Apple GPU/MPS is not implemented. A clean install
has not been tested on every platform.

## Run

```sh
python -u run_methods.py
python -u run_doses.py
```

Run sequentially. Default seeds are **0, 1, 2, 3, 4** (five, not CrabNet's twenty).
CUDA is selected when available; otherwise CPU. Use `--device cuda` to require GPU
or `--device cpu` to require CPU. GPU training also requires GPU-enabled DGL.

```sh
python -u run_methods.py --device cuda --seeds 0 1 --output runs/example
python -u run_doses.py --device cpu --doses 5 15 30 --output runs/example_doses
```

New results go to `runs/methods/` or `runs/doses/`. Nonempty output folders are
refused, not overwritten. This minimal runner does not implement resume.
Per-epoch progress is printed and CSVs are updated during training.
Full dose runs save warm-up and final checkpoints plus synthetic banks; allow
roughly 2 GB of disk space and several GB of RAM for cached graphs.

## What is retained

- Frozen balanced `mp_is_metal` data: **2,014 train, 252 validation, 252 test**,
  each exactly half metallic. IDs, order, labels and compositions match the
  packaged CrabNet splits. ALIGNN additionally needs the included structures.
- Three arms: control (no warm-up), F (fixed structures and random labels),
  R (**the same fixed structures**, new Bernoulli(0.5) labels each minibatch update).
  R does **not** regenerate structures each epoch. It differs from CrabNet's
  input-and-label-refreshed arm. There is no extra U/A construction arm here.
- Synthetic structures preserve lattice and site count. Species are drawn from
  elements 1-100, fractional coordinates from wrapped N(0.5, 0.15^2), with the
  original 200-attempt, 0.5-Angstrom separation policy. True periodic distances
  are checked before training; failures halt rather than discard structures.
- One independent synthetic bank per seed, generated only from training templates.
  Both banks have 2,014 records. Fixed synthetic labels are random, not constrained
  to exactly 50/50. New runs save them in `runs/.../banks/`; the supplied folder
  regenerates them from seeds rather than shipping hundreds of MB of noisy data.
- Native two-class ALIGNN LogSoftmax + NLLLoss; **full-network** updates. Adam
  at 0.001, batch 8, evaluation batch 16, no scheduler, **20 supervised epochs**.
  Optimizer resets at handover; warm-up weights and BatchNorm statistics carry over.
- Shared initialization and matched warm-up/supervised sample orders within seed.
  One warm-up trajectory per method to the maximum dose, with snapshots at each
  dose and a fresh supervised branch from each. Control is trained once per seed.
  The archived dose sweep reused its completed controls; this rerun needs no old weights.
- **252 optimizer updates/pass** (last partial batch retained): 5/15/30 passes
  equal 1,260/3,780/7,560 updates. Methods at 4 passes use 1,008 updates.
- Validation supplies diagnostics only: no checkpoint selection, early stopping
  or temperature fitting. Test inference occurs only after every branch finishes.

## Plot the completed results without training

```sh
python plot.py results/methods
python plot.py results/doses
```

For plotting only, install NumPy, SciPy, pandas, scikit-learn and matplotlib from
the versions in `requirements.txt`; no model libraries are imported by `plot.py`.

`results/` contains the original per-seed metrics (not newly trained results).
Each study has `config.json`, `warmup.csv`, `supervised.csv`, `test.csv`, and
`figures/`. All source metrics are retained. The dose tables store control once,
and the shared warm-up trajectory once, rather than duplicating prefix rows.
Plots appear in `figures/confidence_accuracy_E04.png` for methods, and
`confidence_accuracy_E05.png`, `E15.png`, `E30.png` for doses; PDF copies are included.
`synthetic_bank_nll.png` shows both banks. Run the same plotting command on
`runs/methods` or `runs/doses` for new results.

Confidence/accuracy curves evaluate **real validation materials**, not the noisy
bank or test. Lines are seed means with pointwise 95% Student-t intervals.
Supervised epoch 0 is the handover and is not duplicated. No in-panel explanatory
sentences are drawn. F's on/off-bank NLL uses each bank's fixed labels; R's bank
NLL is the **exact expected loss over independent Bernoulli labels**, not the loss
on a realized training minibatch. Gaps are paired within seed before aggregation.

## Reproducibility and scope

Explicit seed streams cover initialization, synthetic banks, label draws and
sample order. Python, NumPy, PyTorch and DGL are seeded. Deterministic PyTorch
algorithms are required; TF32 is disabled. Unsupported deterministic operations
raise an error instead of silently relaxing the setting. DGL/custom kernels and
different hardware/library builds can still give numerical differences.
Each new run saves settings, source/data SHA-256 hashes and runtime versions.

These are simplified protocol-reproduction scripts, not byte-identical copies of
the original runner. The goal is to replicate the experiment and seed-level
curves, not match checkpoint bytes across devices. No new full GPU run or
GPU-checkpoint-equivalence claim is made. Original source/results are untouched.
Preparation checks matched all ten archived synthetic banks (five seeds, two banks
each), the frozen splits, small native CPU training steps, nested warm-up prefixes,
and the original per-seed metrics and confidence-interval summaries. These checks
do not establish full-run equivalence on other hardware.
Treat these as **exploratory balanced-cohort sensitivity** studies: test materials
had been inspected historically, so this is not a newly untouched test cohort.

Before public upload, choose a code license and verify Matbench/Materials Project
data attribution and redistribution terms; no license or authorship is invented.
