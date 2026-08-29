# Results policy

Only compact, reviewable summaries belong in the Git repository.

- `revision_2026-07-24_pre_consistency_fix/summary/` contains CSV tables reconstructed from the historical revision run tree.
- `validation/` contains small text summaries from exact and nested-estimator smoke checks.
- PyTorch tensors, checkpoints, deterministic state banks, and multi-megabyte pointwise CSVs are distributed separately as versioned release assets with SHA-256 manifests.

The July 24 summaries predate the integrated covariance/loading correction. They must remain visibly versioned and must not be presented as corrected rerun results. See `docs/RESULT_PROVENANCE.md`.
