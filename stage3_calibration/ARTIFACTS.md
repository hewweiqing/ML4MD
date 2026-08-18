# stage3_calibration artifacts

## Included

Package source only (`package/`) — no Stage A run has been executed yet in
any environment (requires `alignn`/`dgl`/CUDA, not available where this was
written; see `package/README.md`).

## Excluded large artifacts

None yet. When Stage A is run on DelftBlue, its output
(`STAGE_A_DIAGNOSTIC_fold*.json` and, on the first real run,
`ALIGNN_MODULE_INVENTORY.json`) should be recorded here in the same
`Artifact | Bytes | SHA-256` table format used by `descriptor_v41/ARTIFACTS.md`
if any output is too large for ordinary Git.
