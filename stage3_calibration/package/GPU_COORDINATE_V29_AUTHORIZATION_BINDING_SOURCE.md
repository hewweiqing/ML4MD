# Sigma-authorization provenance

`stage3_calibration` reuses the exact same authorized
`sigma_cartesian_per_axis_angstrom = 0.04` value that `coordinate_gpu_v4`
uses, per `STAGE3_COORDINATE_SIGMA_AUTHORIZATION_BINDING.json`. This is not
a new prospective selection (no ALIGNN outcomes were consulted) — it is a
byte-identical reuse of the already-frozen decision recorded in
`coordinate_gpu_v4/package/GPU_COORDINATE_V29_AUTHORIZATION_BINDING.json`.

Per your review feedback, this package does **not** reference that file
live across package boundaries (`stage3_calibration` must remain
independently archivable/verifiable). Instead:

- `STAGE3_COORDINATE_SIGMA_AUTHORIZATION_BINDING.json` is a full, self-
  contained duplicate of the same check values (same origin release,
  same authorization SHA-256, same sigma, same selection route, same
  prospective-provenance hash) — it authenticates the literal same
  underlying authorization artifact, not a re-derived one.
- It additionally records `source_binding_sha256` (the SHA-256 of
  `coordinate_gpu_v4/package/GPU_COORDINATE_V29_AUTHORIZATION_BINDING.json`
  as committed in this repo, `9409d90e9fb642ca720289b1997e8b7ced612219404c779521baa7c6b79a4b73`
  at time of writing) purely as an informational cross-audit field — not
  read or enforced by `require_authorized_sigma()` at runtime, just a
  documented trail from this package's binding back to its source.
- At runtime on DelftBlue, `STAGE3_COORDINATE_SIGMA_AUTHORIZATION` must
  point at the same (byte-identical) verified authorization artifact file
  that `ALIGNN_COORDINATE_SIGMA_AUTHORIZATION` points at for
  `coordinate_gpu_v4` — that artifact itself is a runtime-only file, never
  committed to git in either package.
