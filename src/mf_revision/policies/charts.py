"""Differentiable feasible-fiber charts."""
from __future__ import annotations

import torch
import torch.nn.functional as functional


class ControlChart:
    def __init__(
        self,
        risky_dim: int,
        *,
        constraint: str,
        leverage_cap: float = 1.0,
        consumption_rate_min: float = 0.0,
        consumption_rate_max: float | None = None,
    ) -> None:
        self.risky_dim = int(risky_dim)
        self.constraint = str(constraint)
        self.leverage_cap = float(leverage_cap)
        self.consumption_rate_min = float(consumption_rate_min)
        self.consumption_rate_max = (
            None if consumption_rate_max is None else float(consumption_rate_max)
        )

    @property
    def has_consumption(self) -> bool:
        return self.consumption_rate_max is not None

    @property
    def latent_dim(self) -> int:
        portfolio = (
            self.risky_dim
            if self.constraint in {"orthant", "unconstrained"}
            else self.risky_dim + 1
        )
        return portfolio + (1 if self.has_consumption else 0)

    def decode(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected latent dimension {self.latent_dim}, got {latent.shape[-1]}")
        portfolio_dim = self.latent_dim - (1 if self.has_consumption else 0)
        portfolio_latent = latent[:, :portfolio_dim]
        if self.constraint == "orthant":
            risky = functional.softplus(portfolio_latent)
        elif self.constraint == "unconstrained":
            risky = portfolio_latent
        elif self.constraint == "simplex":
            all_weights = torch.softmax(portfolio_latent, dim=-1)
            risky = self.leverage_cap * all_weights[:, : self.risky_dim]
        else:
            raise ValueError(f"Unknown constraint: {self.constraint}")
        if not self.has_consumption:
            return risky
        lower, upper = self.consumption_rate_min, float(self.consumption_rate_max)
        rate = lower + (upper - lower) * torch.sigmoid(latent[:, -1:])
        # In OL mode the rate is frozen, while C=X*c retains the moving-fiber derivative.
        consumption = state[:, 0:1] * rate
        return torch.cat([risky, consumption], dim=-1)
