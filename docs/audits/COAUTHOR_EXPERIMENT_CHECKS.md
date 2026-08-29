# Targeted coauthor experiment checks

This auxiliary audit verifies three revision items from source, resolved configurations, and existing run outputs:

1. the analytical/reference construction used in the Table 6 consumption-cap benchmark;
2. the actual IID, wealth-edge, factor-edge, and stress state definitions;
3. consistency of the nested `M_out x M_in` budgets reported for constrained-factor tables.

## Metadata/source audit without simulation

From the repository root:

```bash
RUN_ROOT=/path/to/runs_paper_full_shift \
  bash scripts/run_coauthor_experiment_checks.sh audit
```

The command cross-checks source code, resolved YAML files, pointwise CSVs, run summaries, and saved evaluation states when available. It writes ignored audit artifacts under `paper/coauthor_checks/`.

## Isolated reruns

The following modes write to `runs_coauthor_rerun/` unless `RERUN_ROOT` is overridden:

```bash
GPU=0 bash scripts/run_coauthor_experiment_checks.sh table6-exact
GPU=0 bash scripts/run_coauthor_experiment_checks.sh table6-all
GPU=0 bash scripts/run_coauthor_experiment_checks.sh lko-budget
GPU=0 bash scripts/run_coauthor_experiment_checks.sh all-targets
```

Learned-policy modes require a valid `paper/legacy_catalog.json`. The wrapper materializes the necessary manifest group before launch and never falls back to an unrequested replacement training run.

Use `FORCE=1` to replace a completed isolated rerun.

## Interpretation

- `PASS`: source, configuration, and available result summaries agree and numerical identities pass;
- `WARN`: an optional saved state or selected run is unavailable;
- `FAIL`: a source identity, state definition, collector value, or nested-budget statement is inconsistent.
