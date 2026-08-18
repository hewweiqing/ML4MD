# Final MUBen temperature-scaling evidence audit

Status: `approved_muben_persistent_final_audit_passed`.

The completed MUBen Uni-Mol evidence audit reports PASS for all six expected cells at release commit `446471d46449ec79cc0f846baa54d7f4a2f90695`. Its source SHA-256 is `f4989a94700990e768cbe595a1399f23dc79e33df2471dd4e12ec32bf5245d43`; the completed evidence archive SHA-256 is `410e96c2257562601646705e4fe6912413bf69d51b9dd9296bbd2c405e4c9c19`.

MUBen exact reproduction used its native optimizer. ALIGNN does not claim that raw unconstrained optimizer as its scientific protocol. The ALIGNN extension requires a one-task shared scalar `T=exp(log_T)`, float64 unweighted validation NLL, and numerical convergence. The v31 derived adapter preserves those semantics and is pinned at SHA-256 `108b3183400725ca53e025c6f0f92690f35069f6fd786091b9d37dc50e90f08b`. It supersedes the v26 source only to declare and enforce a final `log_T` gradient tolerance of `1e-7`; no outer-test outcome was accessed.

Pending, generic, missing, or hash-mismatched approval records fail closed before any outer-test artifact is opened.
