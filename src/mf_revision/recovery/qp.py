"""Exact active-set solvers with a projected-gradient fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np
import torch

Constraint = Literal["unconstrained", "orthant", "simplex"]


@dataclass(slots=True)
class QPSolution:
    control: torch.Tensor
    objective: torch.Tensor
    projected_gradient_residual: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor
    used_fallback: torch.Tensor
    q_used: torch.Tensor
    regularization: torch.Tensor


def _project_simplex_numpy(vector: np.ndarray, cap: float) -> np.ndarray:
    positive = np.maximum(np.asarray(vector, dtype=float), 0.0)
    if positive.sum() <= cap:
        return positive
    ordered = np.sort(positive)[::-1]
    cumulative = np.cumsum(ordered) - cap
    index = np.arange(1, ordered.size + 1)
    valid = ordered - cumulative / index > 0
    rho = np.nonzero(valid)[0][-1]
    threshold = cumulative[rho] / float(rho + 1)
    return np.maximum(positive - threshold, 0.0)


def _project_numpy(vector: np.ndarray, constraint: Constraint, cap: float | None) -> np.ndarray:
    if constraint == "unconstrained":
        return np.asarray(vector, dtype=float)
    if constraint == "orthant":
        return np.maximum(np.asarray(vector, dtype=float), 0.0)
    if constraint == "simplex":
        if cap is None or cap <= 0:
            raise ValueError("simplex constraint requires a positive cap")
        return _project_simplex_numpy(vector, float(cap))
    raise ValueError(f"Unknown constraint {constraint!r}")


def projected_gradient_residual(
    g: torch.Tensor,
    q: torch.Tensor,
    control: torch.Tensor,
    *,
    constraint: Constraint,
    cap: float | None = None,
) -> torch.Tensor:
    """Infinity-norm projected-gradient residual for batched maximization."""
    if g.ndim != 2 or q.ndim != 3 or control.shape != g.shape:
        raise ValueError("Expected g[B,d], q[B,d,d], and control[B,d]")
    residuals: list[float] = []
    for index in range(g.shape[0]):
        gi = g[index].detach().cpu().double().numpy()
        qi = q[index].detach().cpu().double().numpy()
        ui = control[index].detach().cpu().double().numpy()
        lipschitz = max(float(np.linalg.eigvalsh(0.5 * (qi + qi.T)).max()), 1e-14)
        step = 1.0 / lipschitz
        projected = _project_numpy(ui + step * (gi - qi @ ui), constraint, cap)
        residuals.append(float(np.max(np.abs((ui - projected) / step))))
    return torch.as_tensor(residuals, device=g.device, dtype=g.dtype)


def _regularize(q: np.ndarray, floor: float) -> tuple[np.ndarray, float]:
    symmetric = 0.5 * (q + q.T)
    minimum = float(np.linalg.eigvalsh(symmetric).min())
    adjustment = max(float(floor) - minimum, 0.0)
    if adjustment:
        symmetric = symmetric + adjustment * np.eye(symmetric.shape[0])
    return symmetric, adjustment


def _orthant_active_set(
    g: np.ndarray, q: np.ndarray, *, tolerance: float, max_iterations: int
) -> tuple[np.ndarray, int, bool]:
    dimension = g.size
    control = np.zeros(dimension, dtype=float)
    free: set[int] = set(np.flatnonzero(g > tolerance).tolist())
    for iteration in range(1, max_iterations + 1):
        if free:
            index = np.asarray(sorted(free), dtype=int)
            block = q[np.ix_(index, index)]
            try:
                candidate = np.linalg.solve(block, g[index])
            except np.linalg.LinAlgError:
                candidate = np.linalg.lstsq(block, g[index], rcond=None)[0]
            bad = candidate <= tolerance
            if np.any(bad):
                for variable in index[bad]:
                    free.discard(int(variable))
                continue
            control.fill(0.0)
            control[index] = candidate
        else:
            control.fill(0.0)
        gradient = g - q @ control
        inactive = np.asarray(
            [variable for variable in range(dimension) if variable not in free], dtype=int
        )
        if inactive.size:
            violating = inactive[gradient[inactive] > tolerance]
            if violating.size:
                free.add(int(violating[np.argmax(gradient[violating])]))
                continue
        return control, iteration, True
    return control, max_iterations, False


def _simplex_active_set(
    g: np.ndarray,
    q: np.ndarray,
    cap: float,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int, bool]:
    orthant, used, ok = _orthant_active_set(
        g, q, tolerance=tolerance, max_iterations=max_iterations
    )
    if ok and orthant.sum() <= cap + tolerance:
        return orthant, used, True

    dimension = g.size
    free: set[int] = set(np.flatnonzero(orthant > tolerance).tolist())
    if not free:
        free.add(int(np.argmax(g)))
    control = np.zeros(dimension, dtype=float)
    for iteration in range(1, max_iterations + 1):
        index = np.asarray(sorted(free), dtype=int)
        qff = q[np.ix_(index, index)]
        ones = np.ones(index.size)
        system = np.block([[qff, ones[:, None]], [ones[None, :], np.zeros((1, 1))]])
        rhs = np.concatenate([g[index], np.asarray([cap])])
        try:
            solution = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(system, rhs, rcond=None)[0]
        candidate, multiplier = solution[:-1], float(solution[-1])
        bad = candidate <= tolerance
        if np.any(bad):
            removable = index[bad]
            if removable.size >= len(free):
                free = {int(index[np.argmax(candidate)])}
            else:
                for variable in removable:
                    free.discard(int(variable))
            continue
        control.fill(0.0)
        control[index] = candidate
        gradient = g - q @ control
        inactive = np.asarray(
            [variable for variable in range(dimension) if variable not in free], dtype=int
        )
        if inactive.size:
            violating = inactive[gradient[inactive] > multiplier + tolerance]
            if violating.size:
                scores = gradient[violating] - multiplier
                free.add(int(violating[np.argmax(scores)]))
                continue
        return control, iteration, True
    return control, max_iterations, False


def _projected_gradient_fallback(
    g: np.ndarray,
    q: np.ndarray,
    initial: np.ndarray,
    *,
    constraint: Constraint,
    cap: float | None,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int, bool]:
    control = _project_numpy(initial, constraint, cap)
    lipschitz = max(float(np.linalg.eigvalsh(q).max()), 1e-14)
    step = 1.0 / lipschitz
    for iteration in range(1, max_iterations + 1):
        candidate = _project_numpy(control + step * (g - q @ control), constraint, cap)
        residual = np.max(np.abs((control - candidate) / step))
        control = candidate
        if residual <= max(10.0 * tolerance, 1e-12):
            return control, iteration, True
    return control, max_iterations, False


def solve_qp(
    g: torch.Tensor,
    q: torch.Tensor,
    *,
    constraint: Constraint,
    cap: float | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 500,
    curvature_floor: float = 1e-10,
    fallback_iterations: int = 20000,
    raise_on_failure: bool = True,
) -> QPSolution:
    if g.ndim != 2 or q.ndim != 3 or q.shape != (g.shape[0], g.shape[1], g.shape[1]):
        raise ValueError("Expected g[B,d] and q[B,d,d]")
    g_numpy = g.detach().cpu().double().numpy()
    q_numpy = q.detach().cpu().double().numpy()
    controls, q_used, regularization = [], [], []
    iterations, flags, fallbacks = [], [], []
    for state_index in range(g.shape[0]):
        qi, adjustment = _regularize(q_numpy[state_index], curvature_floor)
        gi = g_numpy[state_index]
        if constraint == "unconstrained":
            try:
                control = np.linalg.solve(qi, gi)
            except np.linalg.LinAlgError:
                control = np.linalg.lstsq(qi, gi, rcond=None)[0]
            used, ok, fallback = 1, True, False
        elif constraint == "orthant":
            control, used, ok = _orthant_active_set(
                gi, qi, tolerance=tolerance, max_iterations=max_iterations
            )
            fallback = False
        elif constraint == "simplex":
            if cap is None:
                raise ValueError("simplex constraint requires cap")
            control, used, ok = _simplex_active_set(
                gi, qi, float(cap), tolerance=tolerance, max_iterations=max_iterations
            )
            fallback = False
        else:
            raise ValueError(f"Unknown constraint {constraint!r}")
        if not ok:
            control, extra, ok = _projected_gradient_fallback(
                gi,
                qi,
                control,
                constraint=constraint,
                cap=cap,
                tolerance=tolerance,
                max_iterations=fallback_iterations,
            )
            used += extra
            fallback = True
        controls.append(control)
        q_used.append(qi)
        regularization.append(adjustment)
        iterations.append(used)
        flags.append(ok)
        fallbacks.append(fallback)

    control_tensor = torch.as_tensor(np.stack(controls), device=g.device, dtype=g.dtype)
    q_tensor = torch.as_tensor(np.stack(q_used), device=g.device, dtype=g.dtype)
    residual = projected_gradient_residual(
        g, q_tensor, control_tensor, constraint=constraint, cap=cap
    )
    flag_tensor = torch.as_tensor(flags, device=g.device, dtype=torch.bool)
    if raise_on_failure:
        bad = (~flag_tensor) | (residual > max(100.0 * tolerance, 1e-8))
        if bool(bad.any()):
            index = int(torch.nonzero(bad, as_tuple=False)[0])
            raise RuntimeError(
                f"QP failed at state {index}: flag={flags[index]}, residual={float(residual[index]):.3e}"
            )
    objective = (g * control_tensor).sum(dim=-1) - 0.5 * torch.einsum(
        "bi,bij,bj->b", control_tensor, q_tensor, control_tensor
    )
    return QPSolution(
        control=control_tensor,
        objective=objective,
        projected_gradient_residual=residual,
        iterations=torch.as_tensor(iterations, device=g.device, dtype=torch.int64),
        converged=flag_tensor,
        used_fallback=torch.as_tensor(fallbacks, device=g.device, dtype=torch.bool),
        q_used=q_tensor,
        regularization=torch.as_tensor(regularization, device=g.device, dtype=g.dtype),
    )
