"""Exact and log-barrier scalar consumption recovery."""
from __future__ import annotations

from dataclasses import dataclass
import torch

from mf_revision.models.base import PortfolioModel


@dataclass(slots=True)
class ConsumptionBarrierResult:
    consumption: torch.Tensor
    residual: torch.Tensor
    converged: torch.Tensor
    iterations: torch.Tensor


def consumption_pg_residual(
    model: PortfolioModel,
    state: torch.Tensor,
    lambda_x: torch.Tensor,
    consumption: torch.Tensor,
) -> torch.Tensor:
    lower, upper = model.consumption_bounds(state)
    gradient = model.marginal_utility(consumption) - lambda_x
    curvature = (-model.utility_second_derivative(consumption)).clamp_min(1e-14)
    step = 1.0 / curvature
    projected = torch.minimum(
        torch.maximum(consumption + step * gradient, lower), upper
    )
    return ((consumption - projected) / step).abs().view(-1)


def solve_consumption_barrier(
    model: PortfolioModel,
    state: torch.Tensor,
    lambda_x: torch.Tensor,
    *,
    epsilon: float,
    tolerance: float = 1e-10,
    max_iterations: int = 120,
) -> ConsumptionBarrierResult:
    lower, upper = model.consumption_bounds(state)
    width = (upper - lower).clamp_min(1e-14)
    # Analytic-center start, independent of the DPO action.
    current = lower + 0.5 * width
    converged = torch.zeros(current.shape[0], device=current.device, dtype=torch.bool)
    iterations = torch.zeros(current.shape[0], device=current.device, dtype=torch.int64)
    eps = torch.as_tensor(float(epsilon), device=current.device, dtype=current.dtype)
    boundary = torch.finfo(current.dtype).eps ** 0.5
    for iteration in range(1, int(max_iterations) + 1):
        left = (current - lower).clamp_min(boundary * width)
        right = (upper - current).clamp_min(boundary * width)
        gradient = (
            model.marginal_utility(current)
            - lambda_x
            + eps / left
            - eps / right
        )
        hessian = (
            model.utility_second_derivative(current)
            - eps / left.square()
            - eps / right.square()
        )
        step = -gradient / hessian.clamp_max(-1e-14)
        # Fraction-to-boundary in one dimension.
        positive = step > 0
        maximum = torch.where(
            positive,
            0.995 * right / step.clamp_min(1e-30),
            0.995 * left / (-step).clamp_min(1e-30),
        ).clamp(max=1.0)
        candidate = current + maximum * step
        candidate = torch.minimum(
            torch.maximum(candidate, lower + boundary * width),
            upper - boundary * width,
        )
        residual = gradient.abs().view(-1)
        newly = (~converged) & (residual <= float(tolerance))
        iterations[newly] = iteration
        converged |= newly
        current = torch.where(converged.view(-1, 1), current, candidate)
        if bool(converged.all()):
            break
    final_left = (current - lower).clamp_min(boundary * width)
    final_right = (upper - current).clamp_min(boundary * width)
    final_gradient = (
        model.marginal_utility(current)
        - lambda_x
        + eps / final_left
        - eps / final_right
    ).abs().view(-1)
    iterations[~converged] = int(max_iterations)
    converged |= final_gradient <= max(float(tolerance), 1e-8)
    return ConsumptionBarrierResult(
        consumption=current,
        residual=final_gradient,
        converged=converged,
        iterations=iterations,
    )
