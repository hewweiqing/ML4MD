# Prospective label-free structure preprocessing amendment

This additive operational amendment is made before any ALIGNN scientific run or outer-test outcome access. It does not change folds, labels, models, seeds, optimization, Random2, calibration, or analysis.

The Matbench five-fold design means every structure is supervised training data in four folds and outer-test data in one. v13 continues to permit one global, label-free graph preprocessing pass before model training. The pass reads only each structure dictionary for graph conversion; labels are never retained, indexed, hashed, copied into the cache, or used to select graph parameters. The cache manifest contains structure IDs, structure hashes, complete graph/runtime/release provenance, shard locations, offsets, counts, and graph-file hashes only.

This authorization is limited to deterministic representation preprocessing. Outer-test labels, logits, predictions, metrics, and outcome-dependent decisions remain prohibited until the corresponding cell passes environment, cache, smoke, 100-batch, training, validation-adequacy, Random2, checkpoint, and final temperature-approval gates. Cache creation must record `labels_used_in_graph_construction: false` and exactly one graph construction per cached structure.

This amendment resolves the earlier evidence inconsistency: the A100 environment gate truthfully reports one dataset structure is deserialized, and Stage 1 truthfully reports global label-free structure preprocessing. Neither stage may claim that no dataset or outer-test structure was accessed.
