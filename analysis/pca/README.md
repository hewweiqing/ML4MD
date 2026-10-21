# Cross-model PCA

Reproduce the raw and DIRECT-relative 2D/3D PCA figures for CrabNet, ALIGNN and
ORB from the included checkpoint features. CPU only; no model training needed.

## Run

From this folder, using Python 3.10:

```sh
python -m pip install -r requirements.txt
python analyze.py
```

The 48-page PDF and PNGs are saved in `results/pca_direct/figures/`.
Existing outputs are never overwritten; use `--output results/another_run` to rerun.

## Files

```text
pca/
  analyze.py         # calculate raw and DIRECT-relative PCA
  plot.py            # plot calculated results
  requirements.txt
  README.md
  data/              # required feature inputs and fixed anchor identities
  support/           # methodology, tests, reference checks and author code
```

To replot saved calculations: `python plot.py results/pca_direct --output results/replotted`.
DIRECT is the default; the original anchor sampler remains available with
`--anchors baseline`. Feature inputs, PCA definitions and model checkpoints are unchanged.

See [methodology and optional checks](support/METHODS.md) for representation
definitions, anchor-selection provenance, the independent DIRECT audit and tests.
Keep `data/` and `support/` when uploading or cloning; generated `results/` are ignored.
