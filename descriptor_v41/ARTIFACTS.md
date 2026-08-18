# Descriptor-v41 artifacts

## Included

`results/OOF_ANALYSIS.json` contains the completed OOF analysis and clustered
bootstrap outputs. `results/FINAL_AUDIT.json` records 25 valid cells, 50 trained
branches, 100 readouts, and no failures. The numerical-resolution evidence and
the successful OOF/final-audit logs are also included.

## Excluded large artifacts

The following authoritative local artifacts are excluded from ordinary Git:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `oof_predictions.csv` | 419,627,926 | `65cc032c583d18bfcce6c45a85d36ae5343c5cff622e4b1e7588489a35d82ab7` |
| `alignn_descriptor_v41_final_results.tar.gz` | 82,247,032 | `24bad294760897634cfabbf13734fd303bef0d970844591ce22299d4f798daf5` |

Their current source locations in the research workspace are:

```text
imports/alignn_descriptor_v41_final/extracted/oof_analysis_v41/oof_predictions.csv
imports/alignn_descriptor_v41_final/alignn_descriptor_v41_final_results.tar.gz
```

Publish these separately if row-level reproducibility is required. Do not
recompress or edit an artifact while retaining its old hash.
