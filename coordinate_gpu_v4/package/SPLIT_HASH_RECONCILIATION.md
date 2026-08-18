# Twenty-five-cell split hash reconciliation

`ALL_25_SPLIT_HASHES.json` freezes every `(fold, seed)` cell using the executable policy `random_state=seed`. It contains 25 unique cells, overlap/coverage assertions, ID hashes, validation-label alignment hashes, and official outer-test ID hashes without storing any labels.

The older `STAGE_0B_AUDIT.json` contains one record per fold generated with `random_state=fold`, despite its prose referring to “seed-0.” Its five records therefore match the diagonal cells `(0,0)`, `(1,1)`, `(2,2)`, `(3,3)`, `(4,4)` in the new matrix. v9's conditional seed-0 check would consequently reject fold 1–4 seed 0 and left the other 20 cells unchecked.

v10 does not modify or conceal that historical artifact. It preserves it as evidence, records this reconciliation additively, and makes `ALL_25_SPLIT_HASHES.json` the runtime authority because the frozen paired experiment varies the inner split with the supervised seed. Every training invocation verifies train IDs, validation IDs, and validation labels against its exact fold/seed record before graph loading or optimization.
