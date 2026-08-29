# Revalidation plan

## Gate 0 — software contracts

- deterministic state-indexed Brownian paths;
- deterministic first-step-antithetic/common-future projection paths;
- results invariant to continuation batch size up to floating-point summation;
- actor derivative removed in OL mode;
- moving-fiber derivative retained;
- QP KKT residual at numerical tolerance;
- all runs store configuration, seed, environment, and checkpoint hash.

## Gate 1 — exact scalar Merton

Use the analytical constrained CRRA policy and verify convergence in both the time grid and continuation count for:

- `lambda_x` against `V_x`;
- `P_xx` against `V_xx`;
- `Z_x` against `V_xx sigma_ref`, comparing raw and nested CRN/OLS projections;
- `zeta_x` against zero, reported together with its standard error;
- full-shift versus zero-shift action difference.

A noisy nonzero full-shift action in this test is evidence of estimation noise, not an economic correction.

## Gate 2 — exact predictable-return benchmark

Under the analytical one-factor policy, verify:

- `P_xx` against `V_xx`;
- `P_xy` against `V_xy`;
- `Z_x` against the analytical wealth-row Hessian times state diffusion;
- `zeta_x` against zero;
- decoded policy against the analytical policy.

Run canonical and hedging-relevant calibrations.

## Gate 3 — learned Merton references

For each training budget and seed:

- record policy error, KKT residual, and value diagnostics;
- estimate the finite-reference shift and its signal-to-noise ratio;
- compare full-shift and zero-shift recovered actions;
- repeat over increasing continuation counts.

Value proximity alone must not be used to infer action proximity or a small shift.

## Gate 4 — learned constrained factor references

This is the decisive audit for the paper's current approximation:

- estimate `Z_x` and `zeta_x` rather than setting the shift to zero;
- report shift contribution to the QP linear coefficient;
- report full-vs-zero action RMSE and KKT residuals;
- use independent Brownian draws for downstream policy-value comparison;
- stratify by active, near-switching, and interior regions.

## Gate 5 — paper tables

Only after Gates 0–4 pass should the original tables be recreated at paper budgets. Each table should be generated from a named YAML configuration and a single CLI path; no table-specific hidden projector or differentiation graph is allowed.

## Interpretation rule

The full-shift QP maximizes an estimated local objective. Reusing the same Monte Carlo paths can create an optimistic local gain, especially when true `zeta` is small. Final claims require either analytical truth, a held-out local objective, or independent downstream policy evaluation.
