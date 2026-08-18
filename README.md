# ALIGNN Random2 experiments

This repository-ready bundle consolidates two related ALIGNN experiments on
Matbench `matbench_mp_is_metal` without changing their authoritative source
packages.

## Experiments

### Descriptor intervention (`descriptor_v41`)

The completed descriptor experiment compares Control and Random2-Descriptor,
with raw and MUBen temperature-scaled readouts. Its final OOF consolidation and
audit used the additive v41 package. The underlying 25-cell training outputs
originated in the v26 work root; v36 supplied the primary/MUBen gate and v40
supplied three authorized numerical-resolution exports.

The compact results committed here include:

- `OOF_ANALYSIS.json`;
- `FINAL_AUDIT.json`;
- three v40 numerical-resolution records;
- successful OOF and final-audit logs; and
- the result-package manifest.

The 420 MB row-level prediction table is deliberately excluded from Git. Its
hash and retrieval details are recorded in
[`descriptor_v41/ARTIFACTS.md`](descriptor_v41/ARTIFACTS.md).

### Coordinate intervention (`coordinate_gpu_v4`)

The coordinate experiment compares Control and Random2-Coordinate at the fixed
prospective dose of 0.040 Å per Cartesian axis. The additive v4 package moves
the frozen experiment to A100 CUDA while preserving the CPU permutation stream
used for deterministic Random2 warm-up.

The A100 preflight, corrected smoke, 100-batch profile, manual profile approval,
and fold-0/seed-0 primary gate have passed. The remaining DelftBlue chain was
submitted separately, so no incomplete coordinate result is represented as a
final result in this bundle. See
[`coordinate_gpu_v4/STATUS.md`](coordinate_gpu_v4/STATUS.md).

## Repository layout

```text
descriptor_v41/
  package/                 immutable v41 source package
  results/                 compact audited final results
coordinate_gpu_v4/
  package/                 immutable coordinate-GPU v4 source package
release_metadata/
  descriptor_v41/          builder and archive manifest
  coordinate_gpu_v4/       builder and archive manifest
```

## Integrity checks

Run from the repository root with Python 3.10:

```bash
python verify_packages.py
```

Each package contains its own `PACKAGE_MANIFEST.json`. The release metadata also
records the SHA-256 of the original transfer archive, which is intentionally not
duplicated in Git.

Each `package/` folder's per-version protocol, audit, and correction documents
(the full V5-V41 history) have been consolidated into a single
`*_SUMMARY.md` at the experiment root, and the superseded originals removed;
`PACKAGE_MANIFEST.json` was regenerated to match. `package/` therefore no
longer reproduces the exact `package_aggregate_sha256` recorded in
`release_metadata/*/DELFTBLUE_*_ARCHIVE_MANIFEST_V*.json` for the archive that
was actually built, uploaded, and behaviorally tested on DelftBlue; that
original hash remains preserved unchanged in `release_metadata` as the
bit-exact provenance record. No code, config, or test files were touched by
this trim, only documentation.

## Data policy

Checkpoints, caches, full row-level predictions, transfer archives, and cluster
work directories are excluded. They should be published through an artifact
repository such as 4TU.ResearchData, Zenodo, or a GitHub Release backed by Git
LFS, with the recorded SHA-256 retained.

Before making the repository public, add the intended software license and
dataset citation.
