"""Constant-coefficient CRRA Merton models used by the paper audits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
from scipy.integrate import solve_ivp

from .base import ModelDimensions, PortfolioModel
from .market import (
    legacy_merton_cap_market,
    legacy_merton_short_market,
    load_market_snapshot,
)


@dataclass(slots=True)
class MertonParameters:
    alpha: torch.Tensor
    covariance: torch.Tensor
    loading: torch.Tensor
    reference: torch.Tensor | None = None


class MertonModel(PortfolioModel):
    name = "merton"

    def __init__(self, config: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        d = int(config.get("d", 10))
        self.T = float(config.get("T", 1.5))
        self.rho = float(config.get("rho", 0.0))
        self.gamma = float(config.get("gamma", 2.0))
        self.terminal_weight = float(config.get("terminal_weight", 1.0))
        self.n_steps = int(config.get("n_steps", 20))
        self.r = float(config.get("r", 0.03))
        self.constraint = str(config.get("constraint", "orthant"))
        self.leverage_cap = float(config.get("leverage_cap", 1.0))
        self.has_consumption = bool(config.get("consumption", False))
        self.consumption_rate_min = float(config.get("consumption_rate_min", 0.0))
        self.consumption_rate_max = float(
            config.get("consumption_rate_max", 0.7 if self.has_consumption else 0.0)
        )
        self.wealth_range = tuple(
            float(value) for value in config.get("wealth_range", [0.1, 3.0])
        )
        self.minimum_tau = float(config.get("minimum_tau", 0.0))
        self.minimum_wealth = float(config.get("minimum_wealth", 1e-10))
        self.market_mode = str(
            config.get(
                "market_mode",
                "legacy_cap" if self.has_consumption else "legacy_short",
            )
        ).lower()

        explicit_alpha = config.get("alpha")
        explicit_covariance = config.get("covariance")
        reference: torch.Tensor | None = None
        generated_loading: torch.Tensor | None = None
        snapshot_path = config.get("market_snapshot")
        if snapshot_path:
            snapshot = load_market_snapshot(snapshot_path, device=device, dtype=dtype)
            alpha = snapshot.get("alpha")
            covariance = snapshot.get("Sigma", snapshot.get("covariance"))
            if not torch.is_tensor(alpha) or not torch.is_tensor(covariance):
                raise ValueError("Merton market snapshot must contain alpha and Sigma/covariance")
            if torch.is_tensor(snapshot.get("u_closed_form_unc")):
                reference = snapshot["u_closed_form_unc"]
            elif torch.is_tensor(snapshot.get("reference_control")):
                reference = snapshot["reference_control"].reshape(-1)
            elif torch.is_tensor(snapshot.get("reference")):
                reference = snapshot["reference"].reshape(-1)
        elif explicit_alpha is not None or explicit_covariance is not None:
            if explicit_alpha is None or explicit_covariance is None:
                raise ValueError("model.alpha and model.covariance must be supplied together")
            alpha = torch.as_tensor(explicit_alpha, device=device, dtype=dtype)
            covariance = torch.as_tensor(explicit_covariance, device=device, dtype=dtype)
        else:
            market = dict(config.get("market", {}))
            seed = int(market.get("seed", config.get("market_seed", 42)))
            if self.market_mode in {"legacy_short", "short", "paper_short"}:
                generated = legacy_merton_short_market(
                    d,
                    gamma=self.gamma,
                    seed=seed,
                    device=device,
                    dtype=dtype,
                    vol_range=tuple(market.get("vol_range", [0.18, 0.28])),
                    average_correlation=float(market.get("average_correlation", 0.50)),
                    correlation_jitter=float(market.get("correlation_jitter", 0.06)),
                    alpha_range=tuple(market.get("alpha_range", [0.05, 0.22])),
                    short_fraction=float(market.get("short_fraction", 0.25)),
                    hedge_alpha_range=tuple(market.get("hedge_alpha_range", [-0.03, 0.02])),
                    enforce_shorts=bool(market.get("enforce_shorts", True)),
                    target_negative=int(market.get("target_negative", 1)),
                    max_tries=int(market.get("max_tries", 6)),
                    target_sum=(
                        None if market.get("target_sum") is None else float(market["target_sum"])
                    ),
                    sum_range=tuple(market.get("sum_range", [0.40, 0.75])),
                    sum_bias_exp=float(market.get("sum_bias_exp", 0.7)),
                    rho_div=float(market.get("rho_div", 0.35)),
                    wmax_cap=float(market.get("wmax_cap", 0.70)),
                    short_mass_frac=float(market.get("short_mass_frac", 0.15)),
                    short_count_max_frac=float(market.get("short_count_max_frac", 0.40)),
                    lambda_factor=float(market.get("lambda_factor", 2e-3)),
                )
            elif self.market_mode in {"legacy_cap", "cap", "paper_cap"}:
                generated = legacy_merton_cap_market(
                    d,
                    gamma=self.gamma,
                    seed=seed,
                    device=device,
                    dtype=dtype,
                )
            else:
                raise ValueError(f"Unknown Merton market_mode: {self.market_mode}")
            alpha = generated["alpha"]
            covariance = generated["covariance"]
            reference = generated.get("unconstrained_reference")
            generated_loading = generated.get("loading")

        alpha = torch.as_tensor(alpha, device=device, dtype=dtype)
        covariance = torch.as_tensor(covariance, device=device, dtype=dtype)
        if alpha.shape != (d,) or covariance.shape != (d, d):
            raise ValueError("Market dimensions are incompatible with model.d")
        covariance = 0.5 * (covariance + covariance.T)
        loading = (
            torch.linalg.cholesky(covariance)
            if generated_loading is None
            else generated_loading.to(device=device, dtype=dtype)
        )
        self.params = MertonParameters(
            alpha=alpha,
            covariance=covariance,
            loading=loading,
            reference=None if reference is None else reference.to(device=device, dtype=dtype),
        )
        self.dims = ModelDimensions(
            state=1,
            risky=d,
            brownian=d,
            factor=0,
            control=d + (1 if self.has_consumption else 0),
        )
        self._exact_risky = self._solve_exact_risky()
        self._consumption_grid: tuple[np.ndarray, np.ndarray] | None = None
        if self.has_consumption:
            self._consumption_grid = self._build_value_coefficient_grid()

    @property
    def asset_covariance(self) -> torch.Tensor:
        return self.params.covariance

    @property
    def asset_loading(self) -> torch.Tensor:
        return self.params.loading

    def sample_initial_states(
        self,
        n: int,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        low, high = self.wealth_range
        wealth = low + (high - low) * torch.rand(
            n, 1, generator=generator, device=device, dtype=dtype
        )
        tau = self.minimum_tau + (self.T - self.minimum_tau) * torch.rand(
            n, 1, generator=generator, device=device, dtype=dtype
        )
        return wealth, tau

    def excess_returns(self, state: torch.Tensor) -> torch.Tensor:
        return self.params.alpha.to(state).view(1, -1).expand(state.shape[0], -1)

    def step(
        self, state: torch.Tensor, control: torch.Tensor, dt: torch.Tensor, dW: torch.Tensor
    ) -> torch.Tensor:
        risky, consumption = self.split_control(control)
        wealth = state[:, 0:1].clamp_min(self.minimum_wealth)
        alpha = self.excess_returns(state)
        covariance = self.asset_covariance.to(state)
        loading = self.asset_loading.to(state)
        consumption_rate = torch.zeros_like(wealth) if consumption is None else consumption / wealth
        drift = self.r + (risky * alpha).sum(dim=-1, keepdim=True) - consumption_rate
        variance = torch.einsum("bi,ij,bj->b", risky, covariance, risky).view(-1, 1)
        shock = (risky @ loading * dW).sum(dim=-1, keepdim=True)
        return (
            wealth.log() + (drift - 0.5 * variance) * dt + shock
        ).exp().clamp_min(self.minimum_wealth)

    def state_diffusion(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        risky, _ = self.split_control(control)
        wealth_row = state[:, 0:1] * (risky @ self.asset_loading.to(state))
        return wealth_row.unsqueeze(1)

    def _solve_exact_risky(self) -> torch.Tensor:
        from mf_revision.recovery.qp import solve_qp

        g = self.params.alpha.view(1, -1)
        q = (self.gamma * self.params.covariance).view(
            1, self.dims.risky, self.dims.risky
        )
        return solve_qp(
            g,
            q,
            constraint=self.constraint,
            cap=self.leverage_cap,
            tolerance=1e-12,
            curvature_floor=1e-12,
        ).control[0]

    def exact_risky_weights(
        self, batch: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return self._exact_risky.to(device=device, dtype=dtype).view(1, -1).expand(batch, -1)

    def _portfolio_growth_term(self) -> float:
        risky = self._exact_risky
        return float(
            risky @ self.params.alpha
            - 0.5 * self.gamma * (risky @ self.params.covariance @ risky)
        )

    def _build_value_coefficient_grid(self) -> tuple[np.ndarray, np.ndarray]:
        if abs(self.gamma - 1.0) < 1e-10:
            raise NotImplementedError("Exact consumption benchmark assumes gamma != 1")
        growth = self.r + self._portfolio_growth_term()
        gamma, rho = self.gamma, self.rho
        lower, upper = self.consumption_rate_min, self.consumption_rate_max

        def rhs(_: float, value: np.ndarray) -> np.ndarray:
            coefficient = max(float(value[0]), 1e-14)
            rate = min(max(coefficient ** (-1.0 / gamma), lower), upper)
            consumption = rate ** (1.0 - gamma) - (1.0 - gamma) * coefficient * rate
            return np.array([((1.0 - gamma) * growth - rho) * coefficient + consumption])

        grid = np.linspace(0.0, self.T, 4097)
        solution = solve_ivp(
            rhs,
            (0.0, self.T),
            np.array([self.terminal_weight]),
            t_eval=grid,
            rtol=1e-11,
            atol=1e-13,
        )
        if not solution.success:
            raise RuntimeError(f"Merton coefficient ODE failed: {solution.message}")
        return grid, solution.y[0]

    def _value_coefficient(self, tau: torch.Tensor) -> torch.Tensor:
        if self.has_consumption:
            assert self._consumption_grid is not None
            grid, values = self._consumption_grid
            flat = tau.detach().cpu().numpy().reshape(-1)
            return torch.as_tensor(
                np.interp(flat, grid, values), device=tau.device, dtype=tau.dtype
            ).view_as(tau)
        growth = self.r + self._portfolio_growth_term()
        exponent = ((1.0 - self.gamma) * growth - self.rho) * tau
        return self.terminal_weight * torch.exp(exponent)

    def exact_consumption_rate(self, tau: torch.Tensor) -> torch.Tensor:
        coefficient = self._value_coefficient(tau)
        return coefficient.clamp_min(1e-14).pow(-1.0 / self.gamma).clamp(
            self.consumption_rate_min, self.consumption_rate_max
        )

    def analytical_policy(self):
        from mf_revision.policies.analytic import AnalyticalPolicy
        from mf_revision.policies.factory import build_chart

        chart = build_chart(self)

        def risky_fn(state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
            del tau
            return self.exact_risky_weights(
                state.shape[0], device=state.device, dtype=state.dtype
            )

        consumption_fn = self.exact_consumption_rate if self.has_consumption else None
        return AnalyticalPolicy(chart, risky_fn, consumption_fn)

    def exact_adjoint_fields(self, state: torch.Tensor, tau: torch.Tensor) -> dict[str, torch.Tensor]:
        """Benchmark-specific exact fields.

        In scalar wealth the fixed-latent CRRA identity supplies the second
        adjoint level, so the shifted target is zero.  This is used only as a
        Merton numerical contract, not as a general Markov theorem.
        """
        wealth = state[:, 0:1]
        coefficient = self._value_coefficient(tau)
        lambda_x = wealth.pow(-self.gamma) * coefficient
        p_xx = -self.gamma * wealth.pow(-self.gamma - 1.0) * coefficient
        policy = self.analytical_policy().to(device=state.device, dtype=state.dtype)
        control = policy(state, tau)
        sigma = self.state_diffusion(state, control)
        z_wealth = torch.einsum("bs,bsq->bq", p_xx, sigma)
        return {
            "value": wealth.pow(1.0 - self.gamma) * coefficient / (1.0 - self.gamma),
            "lambda": lambda_x,
            "PXX": p_xx,
            "P_wealth_row": p_xx,
            "Z_wealth": z_wealth,
            "zeta_wealth": torch.zeros_like(z_wealth),
            "control": control,
        }
