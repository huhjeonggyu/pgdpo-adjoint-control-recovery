# Coauthor experiment audit

- Overall: **PASS**
- Repository: `<uploaded-package-test-root>`
- Inspected run root: `runs_paper_full_shift`
- Hard failures: 0
- Warnings: 0

## 1. Table 6 analytical/reference construction

The source and numerical identity checks support the following construction:

- risky reference: exact constrained Merton QP over the configured simplex;
- value coefficient: CRRA coefficient ODE solved by `solve_ivp`;
- consumption reference: proportional rate `A(tau)^(-1/gamma)` clipped to the configured cap;
- exact fields: `lambda_X=A(tau)X^(-gamma)`, `PXX=-gamma*lambda_X/X`, `Z_X=PXX*sigma_X`, and benchmark `zeta_X=0`;
- the exact-policy row still runs the numerical OL-BPTT/shift/recovery pipeline; only the reference actor is analytical.

State source: `deterministic regeneration` (64 states).

| Check | Error | Pass |
|---|---:|:---:|
| `policy_equals_exact_field_control_max_abs` | 0.000000e+00 | yes |
| `consumption_equals_X_times_exact_rate_max_abs` | 0.000000e+00 | yes |
| `PXX_plus_gamma_lambda_over_X_max_abs` | 7.105427e-15 | yes |
| `Z_minus_Psigma_max_abs` | 0.000000e+00 | yes |
| `zeta_zero_max_abs` | 0.000000e+00 | yes |
| `QP_recovered_risky_minus_exact_risky_max_abs` | 1.490127e-15 | yes |
| `scalar_recovered_consumption_minus_exact_max_abs` | 2.220446e-16 | yes |
| `exact_consumption_projected_gradient_residual_max` | 3.789889e-15 | yes |
| `risky_nonnegativity_violation` | 0.000000e+00 | yes |
| `risky_leverage_violation` | 0.000000e+00 | yes |
| `consumption_lower_violation` | 0.000000e+00 | yes |
| `consumption_upper_violation` | 0.000000e+00 | yes |
| `QP_projected_gradient_residual_max` | 4.325356e-16 | yes |

Collector consistency:

- available: `True`
- rows checked: `4`
- maximum absolute difference: `9.237403e-17`
- consistent: `True`

## 2. IID / edge / stress state definitions

| Job | Mode | N | Wealth range observed | Tau range observed | Factor range observed | Saved states | Pass |
|---|---|---:|---|---|---|:---:|:---:|
| `t8_merton_cap_iid` | `iid` | 64 | [0.38749, 2.99483] | [0.018036, 0.988151] | – | no | yes |
| `t8_merton_cap_wealth_edge` | `wealth_edge` | 64 | [0.301211, 2.99959] | [0.018036, 0.988151] | – | no | yes |
| `t8_merton_cap_stress` | `stress` | 64 | [0.3, 3] | [0.01, 1] | – | no | yes |
| `t8_lko_cap_iid` | `iid` | 64 | [0.38749, 2.99483] | [0.0188799, 0.996986] | [-0.2444, 0.247538] | no | yes |
| `t8_lko_cap_wealth_edge` | `wealth_edge` | 64 | [0.302855, 2.99532] | [0.0188799, 0.996986] | [-0.2444, 0.247538] | no | yes |
| `t8_lko_cap_factor_edge` | `factor_edge` | 64 | [0.38749, 2.99483] | [0.0188799, 0.996986] | [-1.5, 1.5] | no | yes |
| `t8_lko_cap_stress` | `stress` | 64 | [0.3, 3] | [0.01, 1] | [-1.5, 1.5] | no | yes |

The code-level definitions are:

- `iid`: the model's canonical initial-state sampler;
- `wealth_edge`: half in the bottom 5% wealth band and half in the top 5% band;
- `factor_edge`: alternating factor values `-1.5, +1.5`;
- `stress`: wealth at the two endpoints, tau at `max(T/100,1e-4)` or `T`, and factors alternating `-1.5, +1.5` when present.

## 3. Nested shifted-input budgets

| Job | Table | d | M | M_out x M_in | Independent outer units | Summary match |
|---|---:|---:|---:|---:|---:|:---:|
| `t6_merton_cap_d2` | 6 | 2 | 8192 | 512 x 8 | 256 | yes |
| `t6_merton_cap_d10` | 6 | 10 | 8192 | 512 x 8 | 256 | yes |
| `t6_merton_cap_d100` | 6 | 100 | 8192 | 1024 x 8 | 512 | yes |
| `t6_merton_cap_d100_exact` | 6 | 100 | 8192 | 1024 x 8 | 512 | yes |
| `t7_affine_canonical` | 7 | 1 | 8192 | 512 x 16 | 256 | yes |
| `t7_affine_hedging` | 7 | 1 | 8192 | 512 x 16 | 256 | yes |
| `t8_merton_cap_iid` | 8 | 100 | 8192 | 1024 x 8 | 512 | yes |
| `t8_merton_cap_wealth_edge` | 8 | 100 | 8192 | 1024 x 8 | 512 | yes |
| `t8_merton_cap_stress` | 8 | 100 | 8192 | 1024 x 8 | 512 | yes |
| `t8_lko_cap_iid` | 8 | 100 | 8192 | 1024 x 8 | 512 | yes |
| `t8_lko_cap_wealth_edge` | 8 | 100 | 8192 | 1024 x 8 | 512 | yes |
| `t8_lko_cap_factor_edge` | 8 | 100 | 8192 | 1024 x 8 | 512 | yes |
| `t8_lko_cap_stress` | 8 | 100 | 8192 | 1024 x 8 | 512 | yes |
| `t12_lko_short_d10` | 12 | 10 | 8192 | 512 x 8 | 256 | yes |
| `t12_lko_short_d50` | 12 | 50 | 8192 | 512 x 8 | 256 | yes |
| `t12_lko_short_d100` | 12 | 100 | 8192 | 1024 x 8 | 512 | yes |

## 4. Issues

No issues were found.

## Interpretation

A PASS means the three coauthor-confirmation items can be documented from the current source, resolved configs, and run summaries without launching a new paper-scale experiment. A separate full-budget rerun is useful only as an independent reproducibility check.
