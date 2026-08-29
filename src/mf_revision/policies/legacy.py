"""Legacy-compatible policy architectures for reusing the paper checkpoints.

The historical code used raw remaining time and the feature order
``(wealth, tau, factors)``.  The classes below intentionally preserve module
names (``net``, ``portfolio_net``, ``consumption_net``, ``net_u``, ``net_c``)
so that old ``state_dict`` files load without rewriting tensors.
"""
from __future__ import annotations

from collections.abc import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from .charts import ControlChart


def _activation(name: str) -> nn.Module:
    key = str(name).lower()
    if key in {"leaky_relu", "leakyrelu"}:
        return nn.LeakyReLU()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


def _mlp(
    input_dim: int,
    output_dim: int,
    hidden: Sequence[int],
    activation: str,
    *,
    xavier: bool,
) -> nn.Sequential:
    modules: list[nn.Module] = []
    previous = int(input_dim)
    for width in hidden:
        modules.extend([nn.Linear(previous, int(width)), _activation(activation)])
        previous = int(width)
    modules.append(nn.Linear(previous, int(output_dim)))
    network = nn.Sequential(*modules)
    if xavier:
        for module in network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.8)
                nn.init.zeros_(module.bias)
    return network


class _LegacyFeatures:
    state_dim: int

    def features(self, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # Historical ordering was X, T-t, Y rather than X, Y, T-t.
        return torch.cat([state[:, 0:1], tau, state[:, 1:]], dim=-1)


class LegacySingleNetworkPolicy(nn.Module, _LegacyFeatures):
    """Historical one-network policy used by Merton-short and LKO-short runs."""

    def __init__(
        self,
        state_dim: int,
        chart: ControlChart,
        *,
        hidden: Sequence[int] = (200, 200),
        activation: str = "leaky_relu",
        xavier: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.chart = chart
        self.net = _mlp(
            self.state_dim + 1,
            chart.latent_dim,
            hidden,
            activation,
            xavier=bool(xavier),
        )

    def latent(self, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(state, tau))

    def decode(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.chart.decode(state, latent)

    def forward(self, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return self.decode(state, self.latent(state, tau))


class LegacyMertonCapPolicy(nn.Module, _LegacyFeatures):
    """Historical split-head Merton consumption-cap policy.

    State-dict names match the legacy ``net_u.*`` and ``net_c.*`` modules.
    """

    def __init__(
        self,
        state_dim: int,
        chart: ControlChart,
        *,
        hidden: Sequence[int] = (200, 200),
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        if not chart.has_consumption or chart.constraint != "simplex":
            raise ValueError("LegacyMertonCapPolicy requires a simplex consumption chart")
        self.state_dim = int(state_dim)
        self.chart = chart
        portfolio_logits = chart.risky_dim + 1
        self.net_u = _mlp(
            self.state_dim + 1,
            portfolio_logits,
            hidden,
            activation,
            xavier=True,
        )
        self.net_c = _mlp(
            self.state_dim + 1,
            1,
            hidden,
            activation,
            xavier=True,
        )

    def latent(self, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        features = self.features(state, tau)
        return torch.cat([self.net_u(features), self.net_c(features)], dim=-1)

    def decode(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.chart.decode(state, latent)

    def forward(self, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return self.decode(state, self.latent(state, tau))


class LegacyFactorCapPolicy(nn.Module, _LegacyFeatures):
    """Historical LKO consumption-cap policy with ``net_u`` and ``net_c`` heads."""

    def __init__(
        self,
        state_dim: int,
        chart: ControlChart,
        *,
        hidden: Sequence[int] = (200, 200),
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        if not chart.has_consumption or chart.constraint != "simplex":
            raise ValueError("LegacyFactorCapPolicy requires a simplex consumption chart")
        self.state_dim = int(state_dim)
        self.chart = chart
        self.net_u = _mlp(
            self.state_dim + 1,
            chart.risky_dim + 1,
            hidden,
            activation,
            xavier=True,
        )
        self.net_c = _mlp(
            self.state_dim + 1,
            1,
            hidden,
            activation,
            xavier=True,
        )

    def latent(self, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        features = self.features(state, tau)
        return torch.cat([self.net_u(features), self.net_c(features)], dim=-1)

    def decode(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.chart.decode(state, latent)

    def forward(self, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return self.decode(state, self.latent(state, tau))
