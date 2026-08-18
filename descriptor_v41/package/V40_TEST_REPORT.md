# v40 test and preservation report

## Scope

The v40 release was built without running training, calibration, inference, pip, network, Slurm, cluster operations or outer-test evaluation. The approved MUBen source was preserved byte-for-byte.

## Source and clean-extraction checks

- package manifest verification: passed;
- Python AST parsing: 84 files passed;
- JSON parsing: 24 files passed;
- `bash -n`: 17 shell/Slurm files passed;
- partition-aware resource audit: all 13 Slurm jobs passed;
- archive path, duplicate-member and normalized-permission audit: passed;
- clean archive extraction and repeated package/static/behavioral verification: passed.

## v40 behavioral coverage

- exact authorized cell set `(1,1)`, `(2,4)`, `(3,1)`;
- exact absolute `log_T` gradient threshold `1e-6`;
- finite strictly positive scalar temperature;
- validation NLL nonincrease;
- exact deterministic repeatability;
- approved MUBen hash binding;
- exact validation-artifact hash binding;
- original false convergence status preserved;
- unauthorized cells, post-outer sentinels, changed artifacts, changed fit metadata, changed module hashes, nondeterminism, invalid temperatures, NLL increase and gradients above threshold fail closed;
- numerical-resolution validation precedes calibration/export, verification and atomic promotion;
- no alternative fitting implementation was added.

## Preservation

- immutable v39 archive SHA-256: `0016e6fb24384e992d86ebc1d94a27582707441f5a4db40e42e44d1b81c43221`;
- immutable v39 package aggregate: `914e610be6b857bbb54b699bcfeb4e425b47529b87cce63793df644125d51ca1`;
- v40 MUBen source SHA-256: `108b3183400725ca53e025c6f0f92690f35069f6fd786091b9d37dc50e90f08b`.

Full runtime pytest and DelftBlue `sbatch --test-only` remain operator-side checks because this local build does not install or invoke the certified DelftBlue runtime or Slurm.
