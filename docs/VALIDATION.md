# Validation status

This page records checks performed while packaging the Git-ready repository. It is not a claim that the full manuscript suite has been rerun after the consistency correction.

## Source and unit checks

Commands:

```bash
python -m compileall -q src tests scripts
PYTHONPATH=src python -m pytest -q
python scripts/verify_covariance_loading_consistency.py
```

Packaging result:

- **22 tests passed** under Python 3.13.5 with NumPy 2.3.5, SciPy 1.17.0, PyTorch 2.10.0+cpu, PyYAML 6.0.3, pandas 2.2.3, and pytest 9.0.2;
- source, test, and script compilation passed;
- the standalone market verifier passed all **23** covariance/loading and cross-covariance checks at tolerance `1e-12`;
- wheel construction and editable installation completed successfully, and the `mf-revision` entry point displayed its command set.

The tests cover:

- deterministic Brownian-bank invariance and antithetic pairing;
- OL versus CL graph-derivative contracts;
- continuation and nested-projection batch invariance;
- exact Merton and exact affine adjoint harvesting;
- nested CRN/OLS Brownian projection;
- unconstrained, orthant, and simplex QP residuals;
- consumption recovery;
- checkpoint compatibility;
- canonical pipeline and independent holdout outputs;
- manifest dependency selection, external per-job configuration loading, and missing-asset reporting;
- Merton-cap and affine covariance/loading identities.

## Manifest checks

The reconstructed paper manifest contains 43 unique jobs and 42 per-job YAML configurations. The following commands complete without private assets:

```bash
PYTHONPATH=src python -m mf_revision.cli suite \
  --manifest paper/paper_suite.yaml --group paper --plan

PYTHONPATH=src python -m mf_revision.cli suite \
  --manifest paper/paper_suite_smoke.yaml --group smoke --plan
```

The full paper plan reports unavailable semantic aliases rather than silently constructing a new policy. The self-contained smoke plan reports no missing assets.

## Exact smoke execution

A self-contained exact Merton smoke pipeline was executed from the cleaned tree as an end-to-end launch check. It produced a complete run directory and was then removed by `make clean`; generated numerical artifacts are intentionally excluded from the source archive.

## Nested-estimator reference audits

The supplied historical exact smoke summaries report that the nested common-random-number OLS projection strongly reduced Brownian-coefficient error relative to raw and centered moment estimators. These values are retained in `results/validation/` as historical validation summaries.

They validate the estimator mechanism at the recorded budgets. They do not replace a corrected full-budget paper rerun.

## Historical paper results not rerun during packaging

The uploaded paper-scale result tree was executed on July 24, 2026. Its run cards report Python 3.10.12, NumPy 2.2.6, PyTorch 2.5.1+cu121, CUDA, and no Git commit. The collector was run against those existing outputs to reconstruct compact table CSVs.

Because those runs predate the integrated covariance/loading correction, affected tables remain labeled `pre_consistency_fix`. See `docs/RESULT_PROVENANCE.md`.
