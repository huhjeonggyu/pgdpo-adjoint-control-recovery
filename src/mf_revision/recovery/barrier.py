"""Double-precision log-barrier audit for quadratic-affine blocks."""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .qp import projected_gradient_residual


@dataclass(slots=True)
class BarrierSolution:
    control: torch.Tensor
    objective: torch.Tensor
    projected_gradient_residual: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor


def solve_barrier_qp(
    g: torch.Tensor,
    q: torch.Tensor,
    *,
    constraint: str,
    cap: float | None = None,
    epsilon: float = 3e-6,
    tolerance: float = 1e-9,
    max_iterations: int = 120,
    fraction_to_boundary: float = 0.995,
    armijo: float = 1e-4,
) -> BarrierSolution:
    if constraint not in {"orthant", "simplex"}:
        raise ValueError("Barrier solver supports orthant and simplex constraints")
    controls, iterations, flags = [], [], []
    for state_index in range(g.shape[0]):
        gi = g[state_index].detach().double()
        qi = 0.5 * (q[state_index].detach().double() + q[state_index].detach().double().T)
        dimension = gi.numel()
        if constraint == "simplex":
            if cap is None:
                raise ValueError("simplex barrier requires cap")
            control = torch.full(
                (dimension,), float(cap) / float(dimension + 1), device=gi.device, dtype=gi.dtype
            )
        else:
            control = torch.full(
                (dimension,), 1.0 / float(max(dimension, 1)), device=gi.device, dtype=gi.dtype
            )

        def value(vector: torch.Tensor) -> torch.Tensor:
            result = gi @ vector - 0.5 * vector @ qi @ vector + epsilon * torch.log(vector).sum()
            if constraint == "simplex":
                result = result + epsilon * torch.log(float(cap) - vector.sum())
            return result

        converged, used = False, 0
        for iteration in range(1, max_iterations + 1):
            used = iteration
            gradient = gi - qi @ control + epsilon / control
            hessian = -qi - epsilon * torch.diag(1.0 / control.square())
            if constraint == "simplex":
                slack = float(cap) - control.sum()
                gradient = gradient - epsilon / slack
                hessian = hessian - epsilon / slack.square() * torch.ones(
                    dimension, dimension, device=control.device, dtype=control.dtype
                )
            direction = torch.linalg.solve(hessian, -gradient)
            decrement_squared = float((gradient @ direction).clamp_min(0.0))
            # For the concave maximization Newton step, decrement_squared is
            # the squared Newton decrement. Its half bounds the remaining
            # local barrier-objective gap.
            if 0.5 * decrement_squared <= tolerance:
                converged = True
                break
            alpha = 1.0
            negative = direction < 0
            if bool(negative.any()):
                alpha = min(
                    alpha,
                    fraction_to_boundary * float(torch.min(-control[negative] / direction[negative])),
                )
            if constraint == "simplex" and float(direction.sum()) > 0:
                alpha = min(
                    alpha,
                    fraction_to_boundary
                    * float((float(cap) - control.sum()) / direction.sum()),
                )
            old_value, slope = value(control), float(gradient @ direction)
            accepted = False
            for _ in range(80):
                candidate = control + alpha * direction
                feasible = bool((candidate > 0).all())
                if constraint == "simplex":
                    feasible = feasible and float(candidate.sum()) < float(cap)
                if feasible and float(value(candidate)) >= float(old_value) + armijo * alpha * slope:
                    control = candidate
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                break
        controls.append(control.to(device=g.device, dtype=g.dtype))
        iterations.append(used)
        flags.append(converged)

    control_tensor = torch.stack(controls)
    objective = (g * control_tensor).sum(dim=-1) - 0.5 * torch.einsum(
        "bi,bij,bj->b", control_tensor, q, control_tensor
    )
    residual = projected_gradient_residual(
        g, q, control_tensor, constraint=constraint, cap=cap
    )
    return BarrierSolution(
        control=control_tensor,
        objective=objective,
        projected_gradient_residual=residual,
        iterations=torch.as_tensor(iterations, device=g.device, dtype=torch.int64),
        converged=torch.as_tensor(flags, device=g.device, dtype=torch.bool),
    )
