"""Affine predictable-return models, including an exact one-factor benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import numpy as np
import torch
from scipy.integrate import solve_ivp

from .base import ModelDimensions, PortfolioModel
from .market import affine_joint_market


@dataclass(slots=True)
class AffineParameters:
    alpha0: torch.Tensor
    alpha_load: torch.Tensor
    kappa: torch.Tensor
    theta: torch.Tensor
    asset_loading: torch.Tensor
    factor_loading: torch.Tensor
    covariance: torch.Tensor
    cross_covariance: torch.Tensor


class AffineFactorModel(PortfolioModel):
    name = "affine_factor"

    def __init__(self, config: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        d, k = int(config.get("d", 1)), int(config.get("k", 1))
        if k <= 0:
            raise ValueError("AffineFactorModel requires k>=1")
        self.T = float(config.get("T", 1.0))
        self.rho = float(config.get("rho", 0.0))
        self.gamma = float(config.get("gamma", 2.0))
        self.terminal_weight = float(config.get("terminal_weight", 1.0))
        self.n_steps = int(config.get("n_steps", 64))
        self.r = float(config.get("r", 0.03))
        self.constraint = str(config.get("constraint", "orthant"))
        self.leverage_cap = float(config.get("leverage_cap", 1.0))
        self.has_consumption = bool(config.get("consumption", False))
        self.consumption_rate_min = float(config.get("consumption_rate_min", 0.0))
        self.consumption_rate_max = float(
            config.get("consumption_rate_max", 0.7 if self.has_consumption else 0.0)
        )
        self.wealth_range = tuple(float(value) for value in config.get("wealth_range", [0.3, 3.0]))
        self.minimum_tau = float(config.get("minimum_tau", max(self.T / 100.0, 1e-4)))
        self.minimum_wealth = float(config.get("minimum_wealth", 1e-10))
        self.exact_one_factor = bool(config.get("exact_one_factor", False))
        self.factor_sampling_mode = str(config.get("factor_sampling_mode", "legacy_positive" if str(config.get("market_mode", "old_affine")) == "old_affine" else "stationary")).lower()

        if self.exact_one_factor:
            if d != 1 or k != 1 or self.has_consumption:
                raise ValueError("exact_one_factor requires d=k=1 and no consumption")
            if self.constraint != "unconstrained":
                raise ValueError("The analytical one-factor policy is unconstrained")
            sigma = float(config.get("sigma", 0.2))
            nu = float(config.get("nu", 0.3))
            corr = float(config.get("corr", 0.1))
            if abs(corr) >= 1.0:
                raise ValueError("|corr| must be <1")
            alpha0 = torch.tensor([float(config.get("a0", 0.0))], device=device, dtype=dtype)
            alpha_load = torch.tensor(
                [[float(config.get("a1", 0.2))]], device=device, dtype=dtype
            )
            kappa = torch.tensor([float(config.get("kappa", 2.0))], device=device, dtype=dtype)
            theta = torch.tensor([float(config.get("theta", 0.2))], device=device, dtype=dtype)
            # W=(W_asset,W_perp), B_Y=corr W_asset+sqrt(1-corr^2)W_perp.
            asset_loading = torch.tensor([[sigma, 0.0]], device=device, dtype=dtype)
            factor_loading = torch.tensor(
                [[nu * corr, nu * math.sqrt(1.0 - corr * corr)]],
                device=device,
                dtype=dtype,
            )
            covariance = asset_loading @ asset_loading.T
            cross_covariance = asset_loading @ factor_loading.T
            self._exact_sigma, self._exact_nu, self._exact_corr = sigma, nu, corr
        else:
            market = affine_joint_market(
                d,
                k,
                seed=int(config.get("market_seed", 11)),
                device=device,
                dtype=dtype,
                mode=str(config.get("market_mode", "old_affine")),
            )
            alpha0 = market["alpha0"]
            alpha_load = market["alpha_load"]
            kappa = market["kappa"]
            theta = market["theta"]
            asset_loading = market["asset_loading"]
            factor_loading = market["factor_loading"]
            covariance = market["covariance"]
            cross_covariance = market["cross_covariance"]

        self.params = AffineParameters(
            alpha0=alpha0,
            alpha_load=alpha_load,
            kappa=kappa,
            theta=theta,
            asset_loading=asset_loading,
            factor_loading=factor_loading,
            covariance=covariance,
            cross_covariance=cross_covariance,
        )
        self.dims = ModelDimensions(
            state=1 + k,
            risky=d,
            brownian=d + k,
            factor=k,
            control=d + (1 if self.has_consumption else 0),
        )
        self._riccati_grid: tuple[np.ndarray, np.ndarray] | None = None
        if self.exact_one_factor:
            self._riccati_grid = self._build_riccati_grid()

    @property
    def asset_covariance(self) -> torch.Tensor:
        return self.params.covariance

    @property
    def asset_loading(self) -> torch.Tensor:
        return self.params.asset_loading

    @property
    def asset_factor_covariance(self) -> torch.Tensor:
        return self.params.cross_covariance

    def excess_returns(self, state: torch.Tensor) -> torch.Tensor:
        factors = state[:, 1:]
        return self.params.alpha0.to(state).view(1, -1) + factors @ self.params.alpha_load.to(state).T

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
        kappa = self.params.kappa.to(device=device, dtype=dtype).clamp_min(1e-8)
        theta = self.params.theta.to(device=device, dtype=dtype)
        factor_volatility = self.params.factor_loading.to(device=device, dtype=dtype).square().sum(dim=-1).sqrt()
        if self.factor_sampling_mode in {"legacy_positive", "positive"}:
            lower = torch.zeros_like(theta)
            upper = theta + 2.0 * factor_volatility
        elif self.factor_sampling_mode in {"stationary", "centered"}:
            standard_deviation = (factor_volatility.square() / (2.0 * kappa)).clamp_min(1e-12).sqrt()
            lower, upper = theta - 2.5 * standard_deviation, theta + 2.5 * standard_deviation
        else:
            raise ValueError(f"Unknown factor_sampling_mode: {self.factor_sampling_mode}")
        factors = lower + (upper - lower) * torch.rand(
            n, self.dims.factor, generator=generator, device=device, dtype=dtype
        )
        tau = self.minimum_tau + (self.T - self.minimum_tau) * torch.rand(
            n, 1, generator=generator, device=device, dtype=dtype
        )
        return torch.cat([wealth, factors], dim=-1), tau

    def step(
        self, state: torch.Tensor, control: torch.Tensor, dt: torch.Tensor, dW: torch.Tensor
    ) -> torch.Tensor:
        risky, consumption = self.split_control(control)
        wealth = state[:, 0:1].clamp_min(self.minimum_wealth)
        factors = state[:, 1:]
        alpha = self.excess_returns(state)
        covariance = self.asset_covariance.to(state)
        asset_loading = self.asset_loading.to(state)
        factor_loading = self.params.factor_loading.to(state)
        consumption_rate = torch.zeros_like(wealth) if consumption is None else consumption / wealth
        drift = self.r + (risky * alpha).sum(dim=-1, keepdim=True) - consumption_rate
        variance = torch.einsum("bi,ij,bj->b", risky, covariance, risky).view(-1, 1)
        wealth_shock = (risky @ asset_loading * dW).sum(dim=-1, keepdim=True)
        next_wealth = (
            wealth.log() + (drift - 0.5 * variance) * dt + wealth_shock
        ).exp().clamp_min(self.minimum_wealth)
        kappa = self.params.kappa.to(state).view(1, -1)
        theta = self.params.theta.to(state).view(1, -1)
        next_factors = factors + kappa * (theta - factors) * dt + dW @ factor_loading.T
        return torch.cat([next_wealth, next_factors], dim=-1)

    def state_diffusion(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        risky, _ = self.split_control(control)
        wealth_row = state[:, 0:1] * (risky @ self.asset_loading.to(state))
        factor_rows = self.params.factor_loading.to(state).view(
            1, self.dims.factor, self.dims.brownian
        ).expand(state.shape[0], -1, -1)
        return torch.cat([wealth_row.unsqueeze(1), factor_rows], dim=1)

    def _build_riccati_grid(self) -> tuple[np.ndarray, np.ndarray]:
        gamma, sigma, nu, corr = (
            self.gamma,
            self._exact_sigma,
            self._exact_nu,
            self._exact_corr,
        )
        kappa, theta = float(self.params.kappa[0]), float(self.params.theta[0])
        a0, a1 = float(self.params.alpha0[0]), float(self.params.alpha_load[0, 0])
        coefficient = (1.0 - gamma) / (2.0 * gamma * sigma * sigma)

        def rhs(_: float, values: np.ndarray) -> np.ndarray:
            a, b, d = values
            h0 = a0 + corr * sigma * nu * b
            h1 = a1 + corr * sigma * nu * d
            da = (
                (1.0 - gamma) * self.r
                - self.rho
                + kappa * theta * b
                + 0.5 * nu * nu * (d + b * b)
                + coefficient * h0 * h0
            )
            db = kappa * theta * d - kappa * b + nu * nu * b * d + 2.0 * coefficient * h0 * h1
            dd = -2.0 * kappa * d + nu * nu * d * d + 2.0 * coefficient * h1 * h1
            return np.array([da, db, dd])

        grid = np.linspace(0.0, self.T, 8193)
        solution = solve_ivp(
            rhs,
            (0.0, self.T),
            np.array([math.log(self.terminal_weight), 0.0, 0.0]),
            t_eval=grid,
            rtol=1e-11,
            atol=1e-13,
        )
        if not solution.success:
            raise RuntimeError(f"Riccati solve failed: {solution.message}")
        return grid, solution.y

    def riccati(self, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._riccati_grid is None:
            raise RuntimeError("No exact Riccati solution configured")
        grid, values = self._riccati_grid
        flat = tau.detach().cpu().numpy().reshape(-1)
        output = [np.interp(flat, grid, values[index]) for index in range(3)]
        tensors = [
            torch.as_tensor(value, device=tau.device, dtype=tau.dtype).view_as(tau)
            for value in output
        ]
        return tensors[0], tensors[1], tensors[2]

    def analytical_policy(self):
        if not self.exact_one_factor:
            raise NotImplementedError("Only exact_one_factor has an analytical policy")
        from mf_revision.policies.analytic import AnalyticalPolicy
        from mf_revision.policies.factory import build_chart

        chart = build_chart(self)

        def risky_fn(state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
            _, b, d = self.riccati(tau)
            factor = state[:, 1:2]
            numerator = self.params.alpha0[0] + self.params.alpha_load[0, 0] * factor
            numerator += self._exact_corr * self._exact_sigma * self._exact_nu * (b + d * factor)
            return numerator / (self.gamma * self._exact_sigma * self._exact_sigma)

        return AnalyticalPolicy(chart, risky_fn)

    def exact_adjoint_fields(self, state: torch.Tensor, tau: torch.Tensor) -> dict[str, torch.Tensor]:
        """Analytical Markov fields for the one-factor Liu benchmark.

        ``lambda`` and ``Z_wealth`` are exact Markov targets.  The scalar
        wealth curvature ``PXX`` is available from the benchmark-specific CRRA
        fixed-latent identity.  ``VXY`` is reported only as a diagnostic for
        the harvested mixed second adjoint; it is not declared to be the exact
        PMP target.
        """
        if not self.exact_one_factor:
            raise NotImplementedError("Only exact_one_factor has exact adjoint fields")
        wealth, factor = state[:, 0:1], state[:, 1:2]
        a, b, d = self.riccati(tau)
        phi = a + b * factor + 0.5 * d * factor.square()
        exponential = torch.exp(phi)
        value = wealth.pow(1.0 - self.gamma) * exponential / (1.0 - self.gamma)
        vx = wealth.pow(-self.gamma) * exponential
        vy = value * (b + d * factor)
        vxx = -self.gamma * wealth.pow(-self.gamma - 1.0) * exponential
        vxy = vx * (b + d * factor)
        value_hessian_row = torch.cat([vxx, vxy], dim=-1)
        policy = self.analytical_policy().to(device=state.device, dtype=state.dtype)
        control = policy(state, tau)
        sigma_state = self.state_diffusion(state, control)
        z_wealth = torch.einsum("bs,bsq->bq", value_hessian_row, sigma_state)
        return {
            "value": value,
            "lambda": torch.cat([vx, vy], dim=-1),
            "PXX": vxx,
            "VXY": vxy,
            "value_hessian_wealth_row": value_hessian_row,
            "Z_wealth": z_wealth,
            "control": control,
            "VXX": vxx,
        }
