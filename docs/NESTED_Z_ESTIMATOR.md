# Nested CRN/OLS estimator for `Z` and `zeta`

## Motivation

The tower-property estimator

\[
\widehat Z_t^{\rm raw}
=\frac1{M\Delta t}\sum_{m=1}^M
\widetilde\lambda_{t+\Delta t}^{(m)}\Delta W_t^{(m)\top}
\]

is valid but can be noisy because the raw next-time derivative still contains all future-shock noise. Dividing by `dt` magnifies that noise. The new estimator approximates the adapted next-step costate before the Brownian projection.

## Outer and inner simulation

For each outer unit, draw an antithetic first-step pair

\[
\Delta W_i^+=-\Delta W_i^-.
\]

This produces two next states `S1+` and `S1-`. For each pair, use the same inner future-shock bank at both next states. Each inner bank is itself antithetic. Define

\[
\bar\lambda_{1,i}^{\pm}
=\frac1L\sum_{\ell=1}^L
D_S^{\rm ol}\widetilde J_{t+\Delta t}
(S_{1,i}^{\pm};\eta_{i\ell}),
\]

and the paired response and design

\[
Y_i=\frac{\bar\lambda_{1,i}^+-\bar\lambda_{1,i}^-}{2},
\qquad
X_i=\Delta W_i^+.
\]

Common random numbers reduce the variance of the difference without changing either marginal conditional expectation.

## Self-normalized Brownian regression

The code fits an OLS slope with an intercept:

\[
Y_i = a + Z_t X_i + e_i.
\]

Equivalently,

\[
\widehat Z_t
=
\left(\sum_i (Y_i-\bar Y)(X_i-\bar X)^\top\right)
\left(\sum_i (X_i-\bar X)(X_i-\bar X)^\top\right)^{-1}.
\]

This uses the realized Brownian Gram matrix instead of replacing it by `dt I`. It is a consistent, self-normalized regression estimator; it is not advertised as exactly unbiased at finite sample size.

## Direct shifted regression

Let

\[
A_t=\widehat P_t\sigma_S(S_t,u_t^{\rm ref}).
\]

Rather than subtracting two separately reported large coefficients after estimation, the code regresses the residual

\[
R_i=Y_i-A_tX_i
\]

directly on `X_i`. The resulting slope is `zeta`, and the reported Brownian coefficient is reconstructed as

\[
\widehat Z_t=A_t+\widehat\zeta_t.
\]

Thus the numerical identity `zeta = Z - P sigma_ref` holds to floating-point precision.

## Standard errors

The OLS slope uses an HC1 componentwise standard error. The main `P` bank and the nested projection bank use separate seeds. Therefore the conditional regression variance and the variance of `P sigma_ref` are combined in quadrature for the reported shift standard error.

## Configuration

```yaml
evaluation:
  z_estimator: nested_crn_ols
  z_outer_paths: 512      # must be even
  z_inner_paths: 16       # one or an even integer
  z_outer_pair_batch: 32
  z_ridge_relative: 1.0e-12
  z_brownian_seed: 722346
  holdout_z_brownian_seed: 1722351
```

For a Brownian dimension `q`, the number of independent outer units is `z_outer_paths/2`. OLS requires this number to exceed `q+1`. For the `n=100` models, the supplied paper configs use 1,024 outer paths and eight inner continuations, giving 512 independent outer units.

## Interpretation

The estimator targets the discrete one-step Brownian projection associated with the chosen time grid. Remaining errors include time discretization, finite inner conditional averaging, finite outer regression, reference-policy error, and ordinary Monte Carlo error. Exact CRRA benchmarks remain the first validation gate before any learned-policy full-shift result is interpreted.
