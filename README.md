# PG-DPO adjoint-to-control recovery

Reproducibility code for **Scalable Pontryagin-Guided Adjoint-to-Control Recovery for Constrained Dynamic Portfolio Choice**.

The package implements the revision-stage audit pipeline used to recover constrained controls from first and second open-loop adjoints:

1. train or load a feedback policy;
2. harvest the first adjoint `lambda` and second adjoint matrix `P` by fixed-latent OL-BPTT;
3. estimate `Z` with a separate nested common-random-number Brownian regression;
4. form the shifted decoder input `zeta = Z - P @ sigma_ref`;
5. recover controls with an exact quadratic-program projection and, when requested, a log-barrier audit;
6. evaluate the recovered controls on an independent holdout Brownian bank.

> [!IMPORTANT]
> The source code in this repository includes the **covariance/loading consistency correction** for the ridged Merton-cap and multi-factor affine market generators. The committed CSV summaries under `results/revision_2026-07-24_pre_consistency_fix/` were reconstructed from the uploaded July 24, 2026 run tree and **predate that correction**. They are retained as archival revision evidence, not as post-correction numerical results. Use `make ridge-rerun` to regenerate every affected job in an isolated output tree.

## Repository contents

- `src/mf_revision/`: reusable Python package and command-line interface.
- `configs/`: self-contained exact and smoke configurations.
- `paper/paper_suite.yaml`: 43-job revision manifest covering Section 5 and appendix experiments.
- `paper/configs/`: 42 path-free per-job configurations; the remaining manifest job is a derived graph comparison.
- `scripts/`: launchers, covariance checks, targeted audits, result collection, and corrected rerun tooling.
- `tests/`: unit and integration tests, including covariance/loading identities.
- `results/`: lightweight CSV summaries only. Large pointwise files and tensors belong in a separate release asset.
- `docs/`: mathematical conventions, estimator notes, provenance, and reproducibility instructions.

## Installation

```bash
git clone https://github.com/huhjeonggyu/pgdpo-adjoint-control-recovery
cd pgdpo-adjoint-control-recovery
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On an offline server with the dependencies already installed:

```bash
python -m pip install --no-build-isolation -e ".[dev]"
```

The package requires Python 3.10 or later, NumPy, SciPy, PyTorch, PyYAML, and pandas.

## Fast verification

These checks require no historical checkpoint:

```bash
make test
make verify-market
make inspect
```

Small end-to-end runs:

```bash
make smoke       # exact plus learned-policy smoke configurations
make exact       # analytical benchmark suite
make zeta        # nested CRN/OLS Z and zeta revalidation
```

Outputs are written under ignored `runs*/` directories. Each completed pipeline records its resolved configuration, environment card, deterministic state/Brownian seeds, adjoints, recovered controls, pointwise diagnostics, and summary JSON/CSV files.

## Paper-scale manifest

The manifest can be inspected without having the private checkpoints:

```bash
make paper-plan
```

Learned-policy paper jobs reuse immutable historical checkpoints and one market snapshot through semantic aliases. Build a machine-local catalog from the read-only legacy archive:

```bash
make legacy-catalog LEGACY_ROOT=/absolute/path/to/legacy_mf_revision
make paper-plan
```

A plan row with an empty `missing_assets` list is runnable. The catalog itself is ignored by Git because it contains local absolute paths.

Run selected groups on a GPU:

```bash
make paper-smoke GPU=0
make paper-exact GPU=0
make paper-section5 GPU=0
make paper-appendix GPU=0
make paper-full GPU=0
make paper-collect
```

The suite automatically includes prerequisite jobs for recovery-only, initialization-audit, and graph-comparison stages. It refuses to materialize or run a learned-policy job when a required catalog alias is absent or points to a missing file; it does not silently retrain a replacement policy.

## Corrected rerun for the consistency issue

The integrated correction makes the Brownian loading used by simulation satisfy the covariance used by the optimizer, while preserving the asset/factor cross covariance in the affine generator. Verify the identities and prepare an isolated rerun of all directly or derivatively affected jobs with:

```bash
make verify-market
make ridge-rerun GPU=0
```

This command:

- verifies 23 covariance/loading and cross-covariance identities at tolerance `1e-12`;
- reruns the affected closure of 28 jobs under `runs_ridge_consistent/`;
- leaves the historical `runs_paper_full_shift/` tree untouched;
- collects corrected CSVs under `paper/collected_ridge_consistent/`.

The original legacy catalog is required because the corrected audit intentionally reuses the same trained policies rather than replacing them with newly trained networks.

## Command-line interface

```bash
mf-revision train    --config CONFIG.yaml
mf-revision harvest  --config CONFIG.yaml [--checkpoint PATH]
mf-revision recover  --config CONFIG.yaml [--adjoints PATH] [--holdout-adjoints PATH]
mf-revision pipeline --config CONFIG.yaml
mf-revision inspect  --config CONFIG.yaml
mf-revision discover-legacy --root LEGACY_ROOT --output paper/legacy_catalog.json
mf-revision suite --manifest paper/paper_suite.yaml --group GROUP --plan
mf-revision collect --manifest paper/paper_suite.yaml --output OUTPUT_DIR
```

All shell launchers also work directly from a source checkout before editable installation.

## Mathematical conventions

- `lambda_`: adapted first open-loop adjoint estimate.
- `p`: adapted second open-loop adjoint estimate; the wealth row is `p_xrow`.
- `z`: one-step Brownian projection of the next first adjoint.
- `zeta`: `z - p @ sigma_ref`; the portfolio decoder uses its wealth component.
- `graph_mode: ol`: actor-state derivatives are removed while explicit chart-state derivatives, such as `C = X c`, are retained.
- `graph_mode: cl`: policy-state derivatives are retained and evaluate the current feedback policy.
- `full_shift`: the complete PMP decoder; `zero_shift` is reported only as an ablation.

An estimated local Hamiltonian improvement is not by itself an out-of-sample policy-value improvement. Exact-policy convergence tests, independent holdout banks, and continuation-budget sweeps remain part of the validation contract.

## Results and provenance

The repository carries manuscript-table summaries small enough for ordinary Git. The full raw result tree is packaged separately because it contains large pointwise CSVs and PyTorch tensors. See:

- `docs/REPRODUCIBILITY.md`
- `docs/RESULT_PROVENANCE.md`
- `results/README.md`
- `docs/MATHEMATICAL_CONTRACT.md`
- `docs/NESTED_Z_ESTIMATOR.md`
- `docs/VALIDATION.md`

Before making the repository public, complete the authorship, citation, checkpoint-distribution, and license decisions in `PUBLIC_RELEASE_CHECKLIST.md`.
