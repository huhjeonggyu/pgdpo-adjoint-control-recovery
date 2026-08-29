# Result provenance and correction boundary

## Source bundles

The cleanup was based on four supplied archives/files dated August 28, 2026:

- a source-code audit archive;
- a full result audit archive;
- a server tree manifest;
- a covariance/loading consistency-fix bundle.

The server manifest stated that the source directory was not a Git repository, and the archived run cards recorded `git_commit: null`. Consequently, the historical results can be tied to their resolved configurations, file hashes, timestamps, and software environment, but not to an original Git commit.

## Historical result date

The paper suite status records execution on July 24, 2026. The lightweight summaries committed under

`results/revision_2026-07-24_pre_consistency_fix/summary/`

were regenerated from that archived run tree by the collector only; no simulation was rerun while producing the summaries.

## Why the historical results are labeled pre-fix

The archived source generated a small ridge-adjusted covariance for two market families but retained a Brownian loading derived from the pre-ridge covariance:

- Merton models with `market_mode: legacy_cap`;
- non-exact affine-factor models using `conservative` or `old_affine` market construction.

Before correction, the covariance implied by the simulator loading differed from the covariance used by the optimizer at relative Frobenius scale roughly `4e-5` to `9e-5` in the supplied verification sweep. The integrated correction rebuilds the loading so that:

- `covariance == loading @ loading.T` for Merton-cap markets;
- `covariance == asset_loading @ asset_loading.T` for affine markets;
- `cross_covariance == asset_loading @ factor_loading.T` remains preserved.

The current verification script checks 23 identities over dimensions 2, 10, 50, and 100 and affine factor dimensions 1, 3, and 5.

## Directly and derivatively affected paper jobs

The corrected closure contains 28 manifest jobs:

- Table 6 Merton-cap rows;
- Table 8/9 Merton-cap and constrained affine rows;
- Table 12 constrained affine rows;
- the appendix graph harvest and graph comparison;
- Table 13 initialization audits that reuse affected runs;
- Table 15 barrier sweeps that reuse affected runs;
- Table 16/17 supplementary multi-factor affine runs.

Tables 3–5 and the exact one-factor affine Table 7 jobs are outside this generator correction.

## Publication rule

Do not merge the historical and corrected summaries under a single unlabeled result directory. Preserve the historical archive for revision traceability and publish corrected results under a new versioned directory after `make ridge-rerun` completes.
