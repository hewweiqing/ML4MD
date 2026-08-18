# v13 local test report

Before archive construction, the inherited suite plus new behavioral tests passed: **89 passed**. It runs again from a clean extraction. v13 coverage includes both partition memory limits and all supported units, all nine scripts, required task/CPU headers, non-exclusive node rejection, exact corrected production headers, mocked all-script test-only success, and rejection of nonzero status, `error:`, allocation failure and per-node warnings.

The helper tests use an injected fake runner and do not execute Slurm. The release builder additionally checks package manifests, Python compilation, JSON parsing, Bash syntax, preservation diff scope, v12/MUBen hashes, clean extraction, archive paths, duplicates, debris and permissions.
