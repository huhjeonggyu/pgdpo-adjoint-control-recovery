"""Model interface for controlled portfolio diffusions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import torch


@dataclass(frozen=True, slots=True)
class ModelDimensions:
    state: int
    risky: int
    brownian: int
    factor: int
    control: int


class PortfolioModel(ABC):
    """Abstract wealth/factor model.

    State convention: ``state[...,0]`` is wealth. The remaining entries are
    Markov factors. Control convention: risky weights first; an optional final
    entry is the consumption amount.
    """

    name: str
    dims: ModelDimensions
    T: float
    rho: float
    gamma: float
    terminal_weight: float
    n_steps: int
    constraint: str
    leverage_cap: float
    has_consumption: bool
    consumption_rate_min: float
    consumption_rate_max: float

    @abstractmethod
    def sample_initial_states(
        self,
        n: int,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return state and remaining horizon ``tau``."""

    @abstractmethod
    def step(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        dt: torch.Tensor,
        dW: torch.Tensor,
    ) -> torch.Tensor:
        """One differentiable simulation step."""

    @abstractmethod
    def state_diffusion(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """State diffusion matrix ``[B,state_dim,brownian_dim]``."""

    @abstractmethod
    def excess_returns(self, state: torch.Tensor) -> torch.Tensor:
        """Risky excess returns ``[B,risky_dim]``."""

    @property
    @abstractmethod
    def asset_covariance(self) -> torch.Tensor:
        """Risky-return covariance matrix."""

    @property
    @abstractmethod
    def asset_loading(self) -> torch.Tensor:
        """Risky-return loading on the independent Brownian basis, ``[d,q]``."""

    @property
    def asset_factor_covariance(self) -> torch.Tensor:
        return torch.empty(
            self.dims.risky,
            self.dims.factor,
            device=self.asset_covariance.device,
            dtype=self.asset_covariance.dtype,
        )

    def utility(self, x: torch.Tensor) -> torch.Tensor:
        safe = x.clamp_min(torch.finfo(x.dtype).tiny)
        if abs(self.gamma - 1.0) < 1e-10:
            return torch.log(safe)
        return safe.pow(1.0 - self.gamma) / (1.0 - self.gamma)

    def marginal_utility(self, x: torch.Tensor) -> torch.Tensor:
        safe = x.clamp_min(torch.finfo(x.dtype).tiny)
        if abs(self.gamma - 1.0) < 1e-10:
            return 1.0 / safe
        return safe.pow(-self.gamma)

    def utility_second_derivative(self, x: torch.Tensor) -> torch.Tensor:
        safe = x.clamp_min(torch.finfo(x.dtype).tiny)
        if abs(self.gamma - 1.0) < 1e-10:
            return -safe.pow(-2.0)
        return -self.gamma * safe.pow(-self.gamma - 1.0)

    def marginal_utility_inverse(self, marginal: torch.Tensor) -> torch.Tensor:
        safe = marginal.clamp_min(torch.finfo(marginal.dtype).tiny)
        if abs(self.gamma - 1.0) < 1e-10:
            return 1.0 / safe
        return safe.pow(-1.0 / self.gamma)

    def running_reward(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        if not self.has_consumption:
            return torch.zeros(state.shape[0], device=state.device, dtype=state.dtype)
        return self.utility(control[:, -1])

    def terminal_reward(self, state: torch.Tensor) -> torch.Tensor:
        return self.terminal_weight * self.utility(state[:, 0])

    def split_control(self, control: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        risky = control[:, : self.dims.risky]
        consumption = control[:, -1:] if self.has_consumption else None
        return risky, consumption

    def local_qp_coefficients(
        self,
        state: torch.Tensor,
        lambda_adapted: torch.Tensor,
        p_wealth_row: torch.Tensor,
        zeta_wealth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``max g'u - 0.5 u'Q u`` from the full shifted input.

        For ``S=(X,Y)``:
        ``g=X[lambda_X alpha + Sigma_asset,Y P_XY + L_asset zeta_X]`` and
        ``Q=X^2(-P_XX) Sigma_asset``.
        """
        x = state[:, 0:1]
        lam_x = lambda_adapted[:, 0:1]
        p_xx = p_wealth_row[:, 0:1]
        alpha = self.excess_returns(state)
        shift = zeta_wealth @ self.asset_loading.to(state).T
        if self.dims.factor:
            p_xy = p_wealth_row[:, 1 : 1 + self.dims.factor]
            hedge = p_xy @ self.asset_factor_covariance.to(state).T
        else:
            hedge = torch.zeros_like(alpha)
        g = x * (lam_x * alpha + hedge + shift)
        covariance = self.asset_covariance.to(state)
        q = (x.square() * (-p_xx)).view(-1, 1, 1) * covariance.view(
            1, self.dims.risky, self.dims.risky
        )
        return g, q

    def consumption_bounds(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lower = self.consumption_rate_min * state[:, 0:1]
        upper = self.consumption_rate_max * state[:, 0:1]
        return lower, upper

    def recover_consumption(
        self, state: torch.Tensor, lambda_adapted: torch.Tensor
    ) -> torch.Tensor | None:
        if not self.has_consumption:
            return None
        unconstrained = self.marginal_utility_inverse(lambda_adapted[:, 0:1])
        lower, upper = self.consumption_bounds(state)
        return torch.minimum(torch.maximum(unconstrained, lower), upper)

    def consumption_local_objective(
        self, consumption: torch.Tensor, lambda_x: torch.Tensor
    ) -> torch.Tensor:
        return self.utility(consumption).view(-1) - (lambda_x * consumption).view(-1)

    def analytical_policy(self) -> Any:
        raise NotImplementedError(f"{self.name} has no analytical policy configured")

    def exact_adjoint_fields(self, state: torch.Tensor, tau: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError(f"{self.name} has no exact adjoint fields")
