"""Analytical policies exposing the same latent/chart contract."""
from __future__ import annotations

from typing import Callable
import torch
import torch.nn as nn

from .charts import ControlChart


class AnalyticalPolicy(nn.Module):
    def __init__(
        self,
        chart: ControlChart,
        risky_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        consumption_rate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.chart = chart
        self.risky_fn = risky_fn
        self.consumption_rate_fn = consumption_rate_fn

    def latent(self, state: torch.Tensor, tau: torch.Tensor) -> dict[str, torch.Tensor]:
        output = {"risky": self.risky_fn(state, tau)}
        if self.consumption_rate_fn is not None:
            output["consumption_rate"] = self.consumption_rate_fn(tau)
        return output

    def decode(self, state: torch.Tensor, latent: dict[str, torch.Tensor]) -> torch.Tensor:
        risky = latent["risky"]
        if "consumption_rate" not in latent:
            return risky
        return torch.cat([risky, state[:, 0:1] * latent["consumption_rate"]], dim=-1)

    def forward(self, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return self.decode(state, self.latent(state, tau))
