"""Neural feedback policy with an explicit latent/chart split."""
from __future__ import annotations

from collections.abc import Sequence
import torch
import torch.nn as nn

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


class NeuralPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        horizon: float,
        chart: ControlChart,
        *,
        hidden: Sequence[int] = (200, 200),
        activation: str = "leaky_relu",
        feature_mode: str = "raw",
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.horizon = float(horizon)
        self.chart = chart
        self.feature_mode = str(feature_mode)
        layers: list[nn.Module] = []
        previous = self.state_dim + 1
        for width in hidden:
            linear = nn.Linear(previous, int(width))
            nn.init.xavier_uniform_(linear.weight, gain=0.8)
            nn.init.zeros_(linear.bias)
            layers.extend([linear, _activation(activation)])
            previous = int(width)
        final = nn.Linear(previous, chart.latent_dim)
        nn.init.xavier_uniform_(final.weight, gain=0.8)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.network = nn.Sequential(*layers)

    def features(self, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if self.feature_mode == "raw":
            transformed = state
        elif self.feature_mode == "log_wealth":
            transformed = torch.cat(
                [state[:, 0:1].clamp_min(1e-12).log(), state[:, 1:]], dim=-1
            )
        else:
            raise ValueError(f"Unknown feature_mode: {self.feature_mode}")
        return torch.cat([transformed, tau / max(self.horizon, 1e-12)], dim=-1)

    def latent(self, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return self.network(self.features(state, tau))

    def decode(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.chart.decode(state, latent)

    def forward(self, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return self.decode(state, self.latent(state, tau))
