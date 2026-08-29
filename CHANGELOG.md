# Changelog

## 0.5.0 — cleaned reproducibility release

- reorganized the revision code as an installable `src/` package;
- added a 43-job manifest and 42 path-free paper configurations;
- separated private checkpoint paths through a machine-local semantic catalog;
- made paper planning possible without private assets while preventing accidental fallback training;
- integrated the covariance/loading consistency correction for ridged Merton-cap and non-exact affine markets;
- added market identity tests and a 23-case standalone verifier;
- added an isolated corrected-rerun workflow covering directly and derivatively affected jobs;
- separated lightweight historical summaries from large raw run outputs;
- removed caches, build metadata, manual backup files, and server-specific paths from the source tree;
- added continuous integration, Git ignore rules, provenance notes, and release guidance.
