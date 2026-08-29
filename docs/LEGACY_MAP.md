# Historical asset migration map

The original revision archive is treated as a read-only asset store. The cleaned repository never imports Python modules from it. Reuse is limited to immutable checkpoint state dictionaries and one market snapshot identified through semantic aliases.

## Catalog workflow

```bash
python -m mf_revision.cli discover-legacy \
  --root /absolute/path/to/legacy_mf_revision \
  --output paper/legacy_catalog.json
```

The generated catalog is machine-local and ignored by Git. `paper/legacy_catalog.example.json` documents the aliases used by `paper/paper_suite.yaml`.

A manifest plan reports an alias as missing when either:

- the alias is absent from the catalog; or
- the catalog path does not exist on the current machine.

Materialization and execution stop before training if a required immutable asset is unavailable.

## Terminology map

| Historical name | Clean-package name |
|---|---|
| `JX` | `lambda_x` |
| `JXX` on a fixed-latent graph | `p_xx` |
| `JXY` on a fixed-latent graph | `p_xy` |
| absent | `z_x` |
| implicit zero shift | `zeta_x` |
| table-specific projector | `recovery.solve_qp` |
| stage-specific shell trees | YAML configuration plus `mf-revision pipeline` |

No historical default projector is trusted implicitly. Borrowing-allowed no-short-sale uses the orthant `u >= 0`; no-borrowing additionally imposes the simplex cap `1' u <= 1`.
