# PCA methodology and optional checks

[Quick start](../README.md)

All commands below are run from `analysis/pca/`, not from this support folder.

CPU-only raw and anchor-relative 2D/static-3D PCA for CrabNet, ALIGNN and ORB.
Checkpoint-derived feature inputs are included. No training, GPU, model weights,
test-set access or original laptop directories are needed.

## Run

Use Python 3.10 and this folder's analysis environment:

```sh
python -m pip install -r requirements.txt
python analyze.py --check-reference
```

**DIRECT is the default.** This refits all 12 PCA models from features and writes
`results/pca_direct/`, including `PCA_explained_variance.csv`, `config.json`,
per-model fits/coordinates and `figures/Cross_model_PCA_only_2D_3D.pdf`.
The PDF has 48 pages: three models x individual/material x raw/relative x
PC1-PC2, PC1-PC3, PC2-PC3 and static PC1-PC2-PC3.

Existing nonempty outputs are refused. The original baseline remains available:

```sh
python analyze.py --anchors baseline --output results/pca_baseline --check-reference
python analyze.py --models crabnet --output results/crabnet_direct --check-reference
python plot.py results/pca_direct --output results/replotted
python -m unittest discover -s support -p test_pca.py -v
```

Use `--no-plots` to calculate without rendering. No interactive plotting is required.

## DIRECT selection and reproducibility

We use the authors' **unchanged DIRECT wrapper**, pinned to PlatonicRep commit
`56cba2354f0879c536dac68d70d509fc82a2c6ae`, with MAML `2025.4.3`:
100 requested anchors, `threshold_init=0.3`, one representative per cluster,
selection seed 0. Actual counts are recorded separately for every model/level.
The authors' adaptive BIRCH threshold rule is retained when fewer than 100
clusters are found; no manual population capping or anchor substitution is used.
The wrapper and its MIT notice are in `support/third_party/`; its SHA-256 is checked
before execution. No custom approximation of DIRECT is used.

For each model/level, selection uses the **complete real training feature
population at the paired Control S20**, never validation/test features or labels.
ORB's previously verified DIRECT selections are reused; the same pinned routine
selects ALIGNN and CrabNet anchors. These are actual observation rows, not
cluster centroids or DIRECT's internal PCA features. Anchors remain in training.

Identities and ordering stay fixed across checkpoints; their **current-state
embeddings** are used at each checkpoint. Normal analysis uses frozen, audited
selections and verifies IDs, counts, hashes, state ordering and original Control
rows. This keeps the five-dependency default offline and prevents numerical
clustering drift from silently changing the analysis.

The complete Control S20 selection populations are also included. To independently
re-run the authors' selector and require identical ordered indices:

```sh
python -m pip install -r support/requirements-direct.txt
python analyze.py --verify-direct-selection --output results/direct_audit --check-reference
```

The optional MAML installation has extra materials-library dependencies; it is
not needed to reproduce figures from frozen selections. Reselection restores
NumPy's global RNG afterward. Selection drift fails explicitly, without fallback
to baseline anchors. No cross-platform bitwise equality is claimed.

## Representations

| Model | Individual | Material descriptor |
| --- | --- | --- |
| ORB | Final backbone atom `node_features`, 256 channels | Mean of all atomic vectors entering the classifier |
| ALIGNN | Final atom nodes before graph pooling, 256 channels | Mean of all atom nodes entering `fc` |
| CrabNet | Valid element tokens before `output_nn.fc_out`, 128 channels | Unweighted masked token mean before the native value/gate readout |

CrabNet has element tokens, not positioned atomic environments: Fe2O3 has two
tokens, not five sites. No extra stoichiometric pooling weights are added.
Scatter subsampling never changes native material pooling.

Relative coordinates at every checkpoint are cosine similarities:

```text
r_j(h) = dot(h, anchor_j) / (norm(h) * norm(anchor_j))
```

Zero/near-zero or nonfinite vectors are rejected. Material-relative features
apply this transformation **after native pooling**, using separate material
anchors; they do not average atomic relative coordinates.

## Relationship to the paper

We adapt [Li and Walsh's framework](https://www.nature.com/articles/s42256-026-01235-7)
and the [authors' implementation](https://github.com/WMD-group/PlatonicRep).
DIRECT selection and the nonzero-vector L2-normalize-then-dot-product cosine
map follow their implementation. Visualization uses **ordinary linear PCA**,
not kernel PCA or nonlinear manifold unfolding. DIRECT's internal standardized,
weighted PCA selects indices only; its coordinates never replace network features.

Our temporal protocol selects anchors from Control S20 and tracks their identities
through checkpoints. Shared training-only PCA supplies common axes. Our material
diagnostic transforms the pooled classifier-input descriptor, unlike the paper's
average of transformed atomic representations. This is a documented adaptation,
not an exact replication of its foundation-MLIP study. Baseline anchors remain
a separately labelled sensitivity option, not the primary analysis.

Suggested Methods wording:

> We adapted Li and Walsh's anchor-relative framework to saved checkpoints.
> For each model and representation level, 100 training observations were
> selected using the authors' pinned DIRECT implementation on the paired
> Control's final supervised features. Anchor identities and ordering were
> fixed across checkpoints, with embeddings recomputed at each state. Relative
> coordinates were cosine similarities to these anchors. Material diagnostics
> transformed the pooled classifier-input descriptor; CrabNet individuals were
> element tokens. A common three-component PCA was fitted to combined training
> references and projected the same validation observations at each state.
> These projections visualize representation evolution, not calibration.

## Fixed PCA and display protocol

- Seed-0 real cohort: 2,014 training and 252 validation materials, each balanced
  by metallicity. No synthetic warm-up bank or test data enters PCA.
- Individual references: one training atom/token per material, seed 8101.
  Material references: all training materials. Concatenate the same reference
  observations across saved states; initialization and Control count once.
- One PCA per model x level x raw/relative view: centered full SVD, three
  components, no whitening/channel scaling, never validation fitting.
- Display: up to eight validation individuals per material, seed 8102, or all
  252 material vectors. Identities, colors, common limits and 3D camera
  (elevation 22, azimuth -55) retain the existing format.
- Axis percentages describe variance of that view's **combined training
  reference**, not accuracy or information content. Relative-view percentages
  are not raw-space variance. Models/views have separate PCA bases.

Main panels show initialization, W5, Control S20 and warmed S20 endpoints.
ORB's shared basis retains all 20 saved W5/W15/W30 states; ALIGNN has six W5/S20
states and CrabNet eight. ORB uses A/B/C persistence arms; ALIGNN uses F/R;
CrabNet uses U-fixed/A-fixed/U-refreshed. They are not a common complete factorial
design. W5 exposure differs: ORB 2,011, ALIGNN 1,260, CrabNet 80 updates in these
archives. Do not attribute differences solely to architecture.

## Files and preserved baseline

- `analyze.py`, `plot.py`: feature-to-PCA calculation and plotting only.
- `data/<model>/`: unchanged baseline reference/anchor/display features and IDs.
- `data/direct/<model>/`: DIRECT IDs, current-state vectors, complete Control
  selection populations and provenance.
- `data/manifest.json`: input hashes, checkpoint/extraction metadata and settings.
- `support/reference/`: unchanged original baseline fits/PDF/provenance.
- `support/reference/direct/`: regression fits for new DIRECT-relative views only.
- `support/VALIDATION.json`: the DIRECT revision's validation; historical baseline
  validation is retained as `support/BASELINE_VALIDATION.json`.
- `results/`: generated locally and ignored by Git.

`--check-reference` compares raw views with original baseline fits in both modes.
Relative views compare with the requested DIRECT or baseline reference. Checks
allow arbitrary PCA axis signs and floating-point tolerance (`1e-8`), not identity
or methodology changes. Raw PCA inputs/fits are unchanged; relative PCA is refitted.

Released subsets cannot support arbitrary alternative anchors without new feature
export. This is not an O3, probe, neighbourhood or training package. New checkpoints
require separate, correctly defined extraction. Include this entire folder
(including data and support) in the repository; exclude results,
environments and model checkpoints.
