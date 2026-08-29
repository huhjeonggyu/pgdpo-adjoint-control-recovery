# Reproducibility guide

The repository separates four reproducibility levels so that a missing private checkpoint cannot be confused with a code failure.

## Level 0: source validation

No external assets or GPU are required.

```bash
make test
make verify-market
make paper-plan
```

Expected behavior:

- Python compilation succeeds;
- the unit/integration suite passes;
- all simulator covariance/loading identities pass at `1e-12`;
- the paper manifest prints a complete plan and explicitly lists unavailable asset aliases.

## Level 1: self-contained numerical smoke tests

```bash
make exact
make smoke
make zeta
make paper-smoke GPU=0
```

The exact configurations use analytical policies. The learned smoke configurations train small policies locally and are intended to detect broken execution paths, not to reproduce manuscript-scale accuracy.

## Level 2: historical revision reproduction

The paper-scale learned-policy jobs require immutable checkpoints from the original server archive. No legacy Python module is imported.

```bash
make legacy-catalog LEGACY_ROOT=/absolute/path/to/legacy_mf_revision
make paper-plan
make paper-section5 GPU=0
make paper-appendix GPU=0
make paper-collect
```

`paper/legacy_catalog.json` maps semantic aliases to local files. It is machine-specific and excluded from Git. `paper/legacy_catalog.example.json` documents the required keys.

The manifest has 43 jobs:

- 32 complete pipelines;
- 6 recovery-only barrier sweeps;
- 3 initialization audits;
- 1 harvest-only closed-loop graph ablation;
- 1 derived graph comparison.

Dependencies are resolved in manifest order. For example, selecting Table 15 first runs the base Table 6 or Table 8 job needed by the recovery sweep.

## Level 3: post-correction rerun

The July 24, 2026 archived run tree predates the covariance/loading correction. The corrected rerun is isolated from the historical output:

```bash
make ridge-rerun GPU=0
```

The selection contains every job whose market generator is directly affected and every downstream audit or comparison that depends on one of those jobs. The generated transient manifest and corrected output roots are ignored by Git.

## Determinism and statistical separation

Supplied configurations specify:

- a global seed;
- a deterministic evaluation-state seed;
- a primary Brownian seed;
- an independent holdout Brownian seed;
- separate primary and holdout Brownian seeds for the nested `Z` regression;
- explicit continuation and pair-batch sizes.

The primary bank estimates adjoints and defines recovered controls. The holdout bank re-estimates the local objective at the same diagnostic states. This prevents in-sample improvement from being reported as independent replication.

Exact bitwise agreement across CUDA architectures is not guaranteed. The run card records the available CUDA devices, Python, NumPy, PyTorch, dtype, and any available Git commit.

## Historical server environment known from run cards

The archived paper-scale run cards consistently report:

- Python 3.10.12;
- NumPy 2.2.6;
- PyTorch 2.5.1+cu121;
- CUDA available;
- `torch.float64` for the numerical pipeline;
- no recorded Git commit (`git_commit: null`).

SciPy, pandas, and PyYAML versions were not recorded in those run cards, so this repository does not invent exact pins for them. The current `pyproject.toml` gives tested minimum versions instead.

## Output contract

A canonical pipeline directory may contain:

- `resolved_config.yaml`;
- `run_card.json`;
- checkpoint hash metadata, when a checkpoint is loaded;
- deterministic evaluation states and Brownian specifications;
- `adjoints.pt` and an independent holdout adjoint file;
- `recovery.pt`;
- pointwise diagnostics;
- adjoint, recovery, and timing summary JSON files.

Large generated tensors and pointwise files are ignored by Git. Publish them as a versioned release asset together with a SHA-256 manifest.
