# Repository cleanup audit

## Supplied material

| Bundle | SHA-256 |
|---|---|
| `mf_code_audit_20260828(1).tar.gz` | `8a0eeb8f6d75c03e5cef4d4c940b1bb09cb8710601b6fb1b3b477272e4dfe7e8` |
| `mf_results_audit_20260828(1).tar.gz` | `3cda9accfb3ffb3c352318da150ac762299626e81f0bc2fbfca16f1ee362581b` |
| `mf_server_manifest_20260828(1).txt` | `936241efb1592669ba1e8c2af7c934defafd614f4ca19c04fbe9f95eb7c1fcc4` |
| `ridge_consistency_fix_bundle(2).tar.gz` | `f6df7771fcf2d053d61252d7897e956bf442b1c3354b9015c352e357348ba396` |

The source archive contained 155 files. The result archive contained 508 files and approximately 135 MB after extraction.

## Removed from the source tree

- 10 `__pycache__` directories;
- 57 compiled `.pyc` files;
- generated `src/mf_revision.egg-info` metadata;
- a manual `barrier.py.before_newton_stop` backup;
- local test caches;
- server-specific absolute paths;
- large raw result trees and PyTorch tensors.

Historical one-off reports were moved out of the repository root into `docs/audits/` or `results/validation/`.

## Added or reconstructed

- installable project metadata and a source-tree CLI;
- `.gitignore`, `.gitattributes`, and GitHub Actions CI;
- 42 path-free paper job configurations and a 43-job manifest;
- machine-local semantic checkpoint catalog workflow;
- portable shell launchers;
- lightweight historical table summaries and SHA-256 manifests;
- result provenance and public-release documentation;
- tests for external config loading, missing asset paths, corrected-job closure, and market covariance/loading identities.

## Consistency correction

The supplied correction was integrated into the canonical market builder rather than retained as a loose patch. Before integration, the standalone sweep failed 23 identities with relative covariance discrepancies at approximately `3.7e-5` to `8.9e-5`. After integration, all identities pass at tolerance `1e-12`, and the ordinary test suite remains green.

## Remaining decisions not inferred during cleanup

- software license;
- final repository authorship and citation metadata;
- whether historical checkpoints and market snapshots may be publicly redistributed;
- final manuscript DOI/arXiv identifier.

These items are listed in `PUBLIC_RELEASE_CHECKLIST.md` rather than being guessed.
