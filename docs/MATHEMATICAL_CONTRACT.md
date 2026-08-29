# Mathematical contract

## State, control, and Brownian basis

The state is `S=(X,Y)` with wealth first. Risky controls are portfolio weights. If consumption is enabled, the final control coordinate is the consumption amount. All model diffusions are expressed in one independent Brownian basis, so

\[
\sigma_S(S,u)\in\mathbb R^{n_S\times q}.
\]

## Two computational graphs

### Feedback graph (`cl`)

The policy is differentiated as a function of the state. For a finite learned policy, conditioning the pathwise derivatives estimates derivatives of that policy's performance.

### Fixed-latent graph (`ol`)

At each rollout state, the actor's latent output is stop-gradiented and then passed through the feasible chart. Thus actor-state derivatives disappear, while structural chart derivatives remain. In particular, for `C=Xc`, the latent rate `c` is frozen but `dC/dX=c` remains.

## Raw sensitivities and adapted projections

For a continuation payoff `J_raw`:

\[
\widetilde\lambda_t=D_S^{ol}J_{raw},\qquad
\widetilde P_t=D_{SS}^{2,ol}J_{raw}.
\]

The code estimates

\[
\lambda_t=E_t[\widetilde\lambda_t],\qquad
P_t=E_t[\widetilde P_t].
\]

The first-interval Brownian projection is

\[
Z_t=
rac{1}{\Delta t}E_t[\lambda_{t+\Delta t}\Delta W_t^	op].
\]

The recommended numerical estimator uses a separate nested projection bank. For outer antithetic increments \(\Delta W_i^+=-\Delta W_i^-\), the code forms plus/minus next states and uses the same inner future-shock bank at both states. The inner-average next-time costates are

\[
ar\lambda_{1,i}^{\pm}
=
rac1L\sum_{\ell=1}^L
D_S^{ol}\widetilde J_{t+\Delta t}(S_{1,i}^{\pm};\eta_{i\ell}).
\]

With

\[
Y_i=
rac{ar\lambda_{1,i}^+-ar\lambda_{1,i}^-}{2},
\qquad X_i=\Delta W_i^+,
\]

the code estimates the Brownian coefficient by a self-normalized OLS slope. Let

\[
A_t=\widehat P_t\sigma_S(S_t,u_t^{ref}).
\]

The shifted input is estimated directly by regressing

\[
R_i=Y_i-A_tX_i
\]

on \(X_i\), and the reported coefficient is reconstructed as

\[
\widehat Z_t=A_t+\widehat\zeta_t.
\]

Thus `zeta = Z - P sigma_ref` holds to floating-point precision. The raw, centered-moment, population-Gram control-variate, and OLS control-variate estimates are retained as audit variants. The main `P` bank and projection bank use separate seeds, so their reported standard errors are combined in quadrature.

## Portfolio recovery

The wealth-component local portfolio block is

\[
\max_{u\in K}
\left\{
 g^\top u-\frac12u^\top Q u
\right\},
\]

with

\[
g=X\left[\lambda^X\alpha(Y)+\Sigma_{SY}P^{XY}+L_{asset}\zeta^X\right],
\qquad
Q=X^2(-P^{XX})\Sigma_S.
\]

The code always computes both:

- the full-shift solution using estimated `zeta`;
- the zero-shift solution using `zeta=0` as an ablation.

If estimated curvature is not positive definite in QP notation, a recorded diagonal regularization is applied. Such a state must not be reported as an exact unregularized recovery without disclosing the adjustment.

## Exact benchmark identities

For exact optimal CRRA selectors under wealth-homogeneous constraints, the wealth row satisfies

\[
P^{X\bullet}=D^2_{X\bullet}V,
\qquad
Z^X=D^2_{X\bullet}V\,\sigma_S^*,
\qquad
\zeta^{X,*}=0.
\]

These identities do not imply that a finite learned reference has zero shift. The exact benchmarks are used to measure the discretization and Monte Carlo errors of the full estimator.

## Independent replication bank

When `evaluation.holdout_continuations>0`, the recovered controls are constructed only from the primary Brownian bank. A second state-indexed bank, with a distinct seed, re-estimates
\((\lambda,P,Z,\zeta)\) at exactly the same diagnostic states. The code then evaluates the already chosen full-shift, zero-shift, and reference actions under the held-out local coefficients. The holdout bank is never used to choose the controls.

This is a local-objective replication audit, not a downstream dynamic policy-value evaluation. A deployable feedback recovery requires a separate adapted adjoint regressor or an online conditional estimator.
