# Manifest-driven paper suite

`paper/paper_suite.yaml` is the canonical specification of the revision experiments. The manifest contains 43 jobs and references 42 path-free YAML files in `paper/configs/`; one additional job compares two previously harvested graph modes and therefore has no standalone model configuration.

## Table map

| Group | Jobs | Purpose |
|---|---:|---|
| `table3` | 2 | no-short-sale Merton scaling |
| `table4` | 2 | exact-policy versus learned-policy adjoint validation |
| `table5` | 9 | three-seed training-budget study |
| `table6` | 4 | Merton consumption-cap recovery, including an exact-policy row |
| `table7` | 2 | exact one-factor affine canonical/hedging cases |
| `table8` | 7 | IID, boundary, and stress audits |
| `table9` | 7 | switching-region summaries derived from Table 8 runs |
| `table12` | 3 | constrained affine scaling in dimensions 10, 50, and 100 |
| `table13` | 3 | initialization audits reusing completed runs |
| `table15` | 6 | barrier-epsilon recovery sweeps |
| `table16` | 3 | internal PG-DPO diagnostics for multi-factor affine runs |
| `table17` | 3 | timing for factor dimensions 1, 3, and 5 |
| `graph_ablation` | 2 | OL/CL harvest comparison |

Jobs may belong to more than one table/group.

## Inspecting the plan

```bash
python -m mf_revision.cli suite \
  --manifest paper/paper_suite.yaml \
  --group paper \
  --plan
```

Planning works without checkpoints. Every row includes `missing_assets` so that a missing policy is visible before an expensive launch.

## Historical checkpoint catalog

The suite uses semantic aliases instead of committing machine-specific paths. Generate the ignored catalog from the original read-only archive:

```bash
python -m mf_revision.cli discover-legacy \
  --root /absolute/path/to/legacy_mf_revision \
  --output paper/legacy_catalog.json
```

`paper/legacy_catalog.example.json` lists the aliases. A learned-policy job refuses to materialize or run if its alias is missing or the target file no longer exists.

## Materialization and execution

```bash
python -m mf_revision.cli suite \
  --manifest paper/paper_suite.yaml \
  --group table6 \
  --materialize

python -m mf_revision.cli suite \
  --manifest paper/paper_suite.yaml \
  --group table6
```

Materialized configs are written to ignored `paper/generated_configs/`. Recovery-only and audit jobs automatically include their prerequisite runs.

Useful wrappers:

```bash
make paper-exact GPU=0
make paper-section5 GPU=0
make paper-appendix GPU=0
make paper-full GPU=0
make paper-collect
```

## Collection

```bash
python -m mf_revision.cli collect \
  --manifest paper/paper_suite.yaml \
  --output paper/collected_full_shift
```

The collector emits manuscript-table CSVs and a collection manifest. It reads completed run summaries; it does not launch simulations.

## Post-correction rerun

The July 24, 2026 archived results predate the covariance/loading consistency correction. After creating `paper/legacy_catalog.json`, run:

```bash
make ridge-rerun GPU=0
```

The script verifies the correction, selects the directly affected jobs plus all changed downstream audits/comparisons, creates an isolated transient manifest, and writes to `runs_ridge_consistent/`.
