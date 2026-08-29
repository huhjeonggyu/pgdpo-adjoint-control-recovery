"""Feedback and fixed-latent open-loop rollout graphs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn

from mf_revision.models.base import PortfolioModel


@dataclass(slots=True)
class RolloutResult:
    payoff: torch.Tensor
    terminal_state: torch.Tensor
    controls: list[torch.Tensor] | None = None
    states: list[torch.Tensor] | None = None


def detach_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {key: detach_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(detach_tree(item) for item in value)
    if isinstance(value, list):
        return [detach_tree(item) for item in value]
    return value


def policy_control(
    policy: nn.Module, state: torch.Tensor, tau: torch.Tensor, *, graph_mode: str
) -> torch.Tensor:
    mode = str(graph_mode).lower()
    if mode == "cl":
        return policy(state, tau)
    if mode != "ol":
        raise ValueError("graph_mode must be ol or cl")
    # Only the actor/selector is frozen. The feasible chart is decoded outside
    # no_grad, so explicit state dependence such as C=X*c remains.
    with torch.no_grad():
        latent = policy.latent(state, tau)  # type: ignore[attr-defined]
    return policy.decode(state, detach_tree(latent))  # type: ignore[attr-defined]


def rollout_payoff(
    model: PortfolioModel,
    policy: nn.Module,
    state0: torch.Tensor,
    tau0: torch.Tensor,
    standard_normals: torch.Tensor,
    *,
    graph_mode: str,
    discount_offset: torch.Tensor | float = 0.0,
    return_path: bool = False,
) -> RolloutResult:
    if standard_normals.ndim != 3:
        raise ValueError("standard_normals must have shape [B,steps,brownian_dim]")
    batch, steps, brownian = standard_normals.shape
    if batch != state0.shape[0] or brownian != model.dims.brownian:
        raise ValueError("Rollout shape mismatch")
    if tau0.shape != (batch, 1):
        raise ValueError("tau0 must have shape [B,1]")
    state = state0
    dt = tau0 / float(steps) if steps else torch.zeros_like(tau0)
    offset = torch.as_tensor(discount_offset, device=state.device, dtype=state.dtype)
    if offset.ndim == 0:
        offset = offset.expand_as(tau0)
    else:
        offset = offset.reshape_as(tau0)
    payoff = torch.zeros(batch, device=state.device, dtype=state.dtype)
    controls: list[torch.Tensor] | None = [] if return_path else None
    states: list[torch.Tensor] | None = [state] if return_path else None
    for index in range(steps):
        tau = tau0 - float(index) * dt
        control = policy_control(policy, state, tau, graph_mode=graph_mode)
        elapsed = offset + float(index) * dt
        if model.has_consumption:
            payoff += (
                torch.exp(-model.rho * elapsed).squeeze(-1)
                * model.running_reward(state, control)
                * dt.squeeze(-1)
            )
        dW = standard_normals[:, index, :] * dt.sqrt()
        state = model.step(state, control, dt, dW)
        if return_path:
            assert controls is not None and states is not None
            controls.append(control)
            states.append(state)
    terminal_discount = torch.exp(-model.rho * (offset + tau0)).squeeze(-1)
    payoff += terminal_discount * model.terminal_reward(state)
    return RolloutResult(payoff=payoff, terminal_state=state, controls=controls, states=states)


@torch.no_grad()
def first_reference_step(
    model: PortfolioModel,
    policy: nn.Module,
    state0: torch.Tensor,
    tau0: torch.Tensor,
    first_standard_normal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dt = tau0 / float(model.n_steps)
    control = policy(state0, tau0)
    next_state = model.step(state0, control, dt, first_standard_normal * dt.sqrt())
    return next_state, control, dt
