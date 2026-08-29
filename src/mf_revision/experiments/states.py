"""Deterministic diagnostic-state samplers for IID, edge, and stress audits."""
from __future__ import annotations

from typing import Any
import torch

from mf_revision.models.base import PortfolioModel


def _uniform(
    shape: tuple[int, ...],
    low: torch.Tensor | float,
    high: torch.Tensor | float,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    random = torch.rand(*shape, generator=generator, device=device, dtype=dtype)
    low_tensor = torch.as_tensor(low, device=device, dtype=dtype)
    high_tensor = torch.as_tensor(high, device=device, dtype=dtype)
    return low_tensor + (high_tensor - low_tensor) * random


def sample_evaluation_states(
    model: PortfolioModel,
    count: int,
    config: dict[str, Any],
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample paper diagnostics while keeping the base model sampler canonical."""
    states, tau = model.sample_initial_states(
        count, generator=generator, device=device, dtype=dtype
    )
    mode = str(config.get("state_mode", "iid")).lower().replace("-", "_")
    if mode in {"iid", "train", "default"}:
        return states, tau

    wealth_low, wealth_high = tuple(float(v) for v in getattr(model, "wealth_range"))
    horizon = float(model.T)
    edge_fraction = float(config.get("edge_fraction", 0.05))
    edge_fraction = min(max(edge_fraction, 1e-4), 0.49)
    half = (count + 1) // 2

    if mode in {"wealth_edge", "wealth"}:
        lower = _uniform(
            (half, 1),
            wealth_low,
            wealth_low + edge_fraction * (wealth_high - wealth_low),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        upper = _uniform(
            (count - half, 1),
            wealth_high - edge_fraction * (wealth_high - wealth_low),
            wealth_high,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        states[:, 0:1] = torch.cat([lower, upper], dim=0)
        return states, tau

    if mode in {"time_edge", "horizon_edge"}:
        low_tau = float(config.get("minimum_tau", max(horizon / 100.0, 1e-4)))
        lower = _uniform(
            (half, 1),
            low_tau,
            low_tau + edge_fraction * (horizon - low_tau),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        upper = _uniform(
            (count - half, 1),
            horizon - edge_fraction * (horizon - low_tau),
            horizon,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        return states, torch.cat([lower, upper], dim=0)

    if mode in {"factor_edge", "factor"}:
        if model.dims.factor <= 0:
            raise ValueError("factor_edge requires a factor model")
        values = config.get("factor_values", [-1.5, 1.5])
        values_tensor = torch.as_tensor(values, device=device, dtype=dtype).flatten()
        if values_tensor.numel() < 2:
            raise ValueError("factor_values must contain at least two values")
        pattern = values_tensor[torch.arange(count, device=device) % values_tensor.numel()]
        states[:, 1:] = pattern.view(-1, 1).expand(-1, model.dims.factor)
        return states, tau

    if mode in {"stress", "joint_stress"}:
        wealth_pattern = torch.where(
            (torch.arange(count, device=device) % 2).view(-1, 1) == 0,
            torch.full((count, 1), wealth_low, device=device, dtype=dtype),
            torch.full((count, 1), wealth_high, device=device, dtype=dtype),
        )
        states[:, 0:1] = wealth_pattern
        low_tau = float(config.get("minimum_tau", max(horizon / 100.0, 1e-4)))
        tau = torch.where(
            (torch.arange(count, device=device) % 4 < 2).view(-1, 1),
            torch.full((count, 1), low_tau, device=device, dtype=dtype),
            torch.full((count, 1), horizon, device=device, dtype=dtype),
        )
        if model.dims.factor:
            values = torch.as_tensor(
                config.get("factor_values", [-1.5, 1.5]),
                device=device,
                dtype=dtype,
            ).flatten()
            pattern = values[torch.arange(count, device=device) % values.numel()]
            states[:, 1:] = pattern.view(-1, 1).expand(-1, model.dims.factor)
        return states, tau

    if mode == "custom":
        if "wealth_range" in config:
            low, high = map(float, config["wealth_range"])
            states[:, 0:1] = _uniform(
                (count, 1), low, high,
                generator=generator, device=device, dtype=dtype,
            )
        if "tau_range" in config:
            low, high = map(float, config["tau_range"])
            tau = _uniform(
                (count, 1), low, high,
                generator=generator, device=device, dtype=dtype,
            )
        if model.dims.factor and "factor_range" in config:
            low, high = config["factor_range"]
            states[:, 1:] = _uniform(
                (count, model.dims.factor), low, high,
                generator=generator, device=device, dtype=dtype,
            )
        return states, tau

    raise ValueError(f"Unknown evaluation.state_mode: {mode}")
