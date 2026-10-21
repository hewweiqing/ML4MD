# CrabNet warm-up experiments
Structure:

| File | What it does |
| --- | --- |
| `run_methods.py` | Compare Control, fixed U, fixed A and refreshed U at 16 warm-up epochs. Also contains the shared training functions. |
| `run_doses.py` | Run those same methods at 5, 15 and 30 warm-up epochs. Imports the shared functions rather than duplicating them. |
| `plot.py` | Draw the four-panel confidence–accuracy figures from saved CSVs. No PyTorch/GPU needed for plotting. |

## Install

Use Python **3.10** in a fresh environment:

```sh
python -m pip install -r requirements.txt
```

`requirements.txt` lists the direct dependencies. `environment.txt` records the
original environment's `pip list --format=freeze` (version pins without local
Conda build paths) for reference; it is not a portable lockfile.
Install from `requirements.txt`, **not** `environment.txt`: the latter also lists
unrelated packages from the author's laptop. Matbench itself is not needed because
the curated data are included.
The code uses relative paths and works with CPU or CUDA without Windows-specific
commands. CUDA requires an NVIDIA GPU and a compatible PyTorch installation.
The original run used PyTorch 2.0.1+cu118; to install that CUDA build:

```sh
python -m pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## Run

Run these commands from this folder:

```sh
python -u run_methods.py
python -u run_doses.py
```

These commands automatically select CUDA when available, otherwise CPU. Use
`--device cuda` to require GPU training or `--device cpu` to require CPU. `-u`
prints progress immediately. Run the two experiments sequentially. To choose
seeds or a different fixed dose:

```sh
python run_methods.py --warmup-epochs 16 --seeds 0 1 2 --output runs/example
```

Defaults: seeds **0–19**, **20 supervised epochs**, effective batch **128**.
Each seed has one control and three warm-up methods. Fixed U/A share random labels
and use microbatch 16. Refreshed U replaces inputs and labels each epoch and keeps
the original jitter and microbatch 128. U preserves distinct-element counts;
A samples atoms with replacement from reduced stoichiometric atom counts.
No atom counts are silently capped.

The dose sweep uses one 30-pass trajectory per seed/method and snapshots at 5, 15,
30. These equal 80, 240, 480 warm-up updates. Warm-up diagnostics evaluate the real
validation set and both trained-on and independent synthetic banks.
Supervision retains BCE, the native optimizer/scheduler/SWA policy, and the frozen
element embedding. Test evaluation happens only after all training finishes. No TS.

New results go to `runs/methods/` or `runs/doses/`. Existing output folders are never
overwritten. This minimal runner does not implement interrupted-run resumption.
It saves final weights for the global test gate, so allow roughly 12 GB for a full
dose sweep. New runs train one control per seed; the archived study reused controls.

## Plot results from a completed run

After training finishes, run these commands from the `crabnet/` folder:

```sh
python plot.py runs/doses
python plot.py runs/methods
```

Use the output directory corresponding to the experiment you completed.
Archived reference results are not included in this repository.

Completed runs contain `config.json`, `warmup.csv`, `supervised.csv`,
`test.csv`, `checkpoints/`, and `figures/`.

Figures show Control / fixed U / fixed A / refreshed U, from warm-up
to supervision, with seed means and pointwise 95% t confidence intervals.
Confidence and accuracy curves use real validation data.
`test.csv` contains the separate final test metrics.

The shared dataset is in the repository root at `../data/`.
It contains balanced `mp_is_metal` splits: 2,014 training,
252 validation, and 252 test materials, exactly half metallic in each.
Synthetic bank labels remain Bernoulli(0.5), not forced to exact balance.

## Reproducibility

Seeds cover Python, NumPy, PyTorch, sample order and synthetic banks. Named streams
use stable SHA-256 seed derivation, not Python's process-dependent `hash()`.
Diagnostics preserve RNG state; every branch receives the same supervised seed.
Deterministic PyTorch algorithms are enabled and fail explicitly if unsupported.
If a device reports an unsupported deterministic operation, use `--device cpu`;
the script does not silently relax determinism or change the experiment.
Each new run records its seeds, settings, data/embedding hashes and runtime versions
in `config.json`. Use the same dependencies/device for the closest repeatability.
Different hardware or library builds can still produce numerical differences.
Compare seed-level metrics and mean/CI curves rather than checkpoint-file
hashes across devices.

These are simplified rerun scripts, not the original source snapshot.
New runs train one control per seed; the archived study reused controls.
The repository reorganization does not establish that the original
experimental results have been reproduced.

Before publishing, choose a code license and check Matbench/Materials Project
dataset attribution and redistribution terms. No license or author citation has
been invented here.
