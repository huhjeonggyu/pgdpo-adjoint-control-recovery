# v0.4 paper-suite changes

- Reuses the frozen manuscript checkpoints through strict legacy-compatible policy classes.
- Reproduces the historical Merton, Merton-cap, constrained LKO, and supplementary affine market generators.
- Makes nested CRN/OLS residual-shift estimation the paper-scale default.
- Uses the full shifted input in QP-PGDPO and B-PGDPO; zero shift is a paired ablation.
- Adds scalar consumption KKT/barrier recovery and full joint diagnostics.
- Adds deterministic IID/edge/stress state samplers and independent holdout Brownian banks.
- Adds a dependency-aware manifest covering Section 5 and Appendix E.
- Adds checkpoint discovery, initialization and graph audits, table collection, and configurable-GPU make targets.
- Uses historical DPO checkpoints when supplied; the v0.5 manifest layer now refuses missing paper assets instead of silently replacing them.
