"""Full-shift recovery, zero-shift ablation, consumption, and barrier audits."""
from __future__ import annotations

from typing import Any
import torch

from mf_revision.models.base import PortfolioModel
from mf_revision.types import AdjointEstimate, RecoveryResult
from .barrier import solve_barrier_qp
from .consumption import (
    consumption_pg_residual,
    solve_consumption_barrier,
)
from .qp import projected_gradient_residual, solve_qp


def _portfolio_objective(
    g: torch.Tensor, q: torch.Tensor, control: torch.Tensor
) -> torch.Tensor:
    return (g * control).sum(dim=-1) - 0.5 * torch.einsum(
        "bi,bij,bj->b", control, q, control
    )


def _consumption_objective(
    model: PortfolioModel,
    consumption: torch.Tensor | None,
    lambda_x: torch.Tensor,
) -> torch.Tensor:
    if consumption is None:
        return torch.zeros(lambda_x.shape[0], device=lambda_x.device, dtype=lambda_x.dtype)
    return model.consumption_local_objective(consumption, lambda_x)


def _joint_objective(
    model: PortfolioModel,
    g: torch.Tensor,
    q: torch.Tensor,
    risky: torch.Tensor,
    consumption: torch.Tensor | None,
    lambda_x: torch.Tensor,
) -> torch.Tensor:
    return _portfolio_objective(g, q, risky) + _consumption_objective(
        model, consumption, lambda_x
    )


def _combine_control(
    risky: torch.Tensor, consumption: torch.Tensor | None
) -> torch.Tensor:
    return risky if consumption is None else torch.cat([risky, consumption], dim=-1)


def _split_reference(
    model: PortfolioModel, control: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return model.split_control(control)


def _summary_exact_controls(
    model: PortfolioModel,
    estimate: AdjointEstimate,
    *,
    full_control: torch.Tensor,
    zero_control: torch.Tensor,
    barrier_control: torch.Tensor | None,
    reference_control: torch.Tensor,
) -> dict[str, torch.Tensor]:
    try:
        fields = model.exact_adjoint_fields(estimate.states, estimate.tau)
    except (NotImplementedError, RuntimeError):
        return {}
    exact = fields.get("control")
    if exact is None or not torch.is_tensor(exact):
        return {}
    exact = exact.to(device=full_control.device, dtype=full_control.dtype)
    risky = model.dims.risky
    diagnostics: dict[str, torch.Tensor] = {
        "reference_action_l2_to_exact": torch.linalg.norm(reference_control - exact, dim=-1),
        "full_action_l2_to_exact": torch.linalg.norm(full_control - exact, dim=-1),
        "zero_action_l2_to_exact": torch.linalg.norm(zero_control - exact, dim=-1),
        "reference_risky_l2_to_exact": torch.linalg.norm(
            reference_control[:, :risky] - exact[:, :risky], dim=-1
        ),
        "full_risky_l2_to_exact": torch.linalg.norm(
            full_control[:, :risky] - exact[:, :risky], dim=-1
        ),
        "zero_risky_l2_to_exact": torch.linalg.norm(
            zero_control[:, :risky] - exact[:, :risky], dim=-1
        ),
    }
    if model.has_consumption:
        diagnostics.update(
            {
                "reference_consumption_abs_to_exact": (
                    reference_control[:, -1] - exact[:, -1]
                ).abs(),
                "full_consumption_abs_to_exact": (
                    full_control[:, -1] - exact[:, -1]
                ).abs(),
                "zero_consumption_abs_to_exact": (
                    zero_control[:, -1] - exact[:, -1]
                ).abs(),
            }
        )
    if barrier_control is not None:
        diagnostics["barrier_action_l2_to_exact"] = torch.linalg.norm(
            barrier_control - exact, dim=-1
        )
        diagnostics["barrier_risky_l2_to_exact"] = torch.linalg.norm(
            barrier_control[:, :risky] - exact[:, :risky], dim=-1
        )
        if model.has_consumption:
            diagnostics["barrier_consumption_abs_to_exact"] = (
                barrier_control[:, -1] - exact[:, -1]
            ).abs()
    return diagnostics


def recover_controls(
    model: PortfolioModel,
    estimate: AdjointEstimate,
    config: dict[str, Any],
    *,
    holdout_estimate: AdjointEstimate | None = None,
) -> RecoveryResult:
    """Recover the full-shift and zero-shift controls at diagnostic states.

    The risky block is solved as a QP.  When consumption is present, the exact
    KKT clipping is coupled only through the common first-adjoint level and is
    therefore solved separately.  B-PGDPO uses the same full shifted input and
    applies the log barrier to both the risky and scalar consumption blocks.
    """
    p_row = estimate.p_xrow
    g_full, q_raw = model.local_qp_coefficients(
        estimate.states, estimate.lambda_, p_row, estimate.zeta_x
    )
    g_zero, _ = model.local_qp_coefficients(
        estimate.states, estimate.lambda_, p_row, torch.zeros_like(estimate.zeta_x)
    )
    tolerance = float(config.get("qp_tolerance", 1e-10))
    curvature_floor = float(config.get("curvature_floor", 1e-10))
    max_iterations = int(config.get("qp_max_iterations", 500))
    full = solve_qp(
        g_full,
        q_raw,
        constraint=model.constraint,
        cap=model.leverage_cap,
        tolerance=tolerance,
        max_iterations=max_iterations,
        curvature_floor=curvature_floor,
    )
    q = full.q_used
    zero = solve_qp(
        g_zero,
        q,
        constraint=model.constraint,
        cap=model.leverage_cap,
        tolerance=tolerance,
        max_iterations=max_iterations,
        curvature_floor=0.0,
    )

    # The shift affects the portfolio block.  Full and zero decoders use the
    # same first-adjoint level for the exact scalar consumption KKT clipping.
    consumption = model.recover_consumption(estimate.states, estimate.lambda_)
    full_control = _combine_control(full.control, consumption)
    zero_control = _combine_control(zero.control, consumption)

    barrier_control: torch.Tensor | None = None
    barrier_solution = None
    consumption_barrier = None
    barrier_enabled = bool(config.get("barrier", config.get("barrier_epsilon") is not None))
    barrier_epsilon = float(config.get("barrier_epsilon", 3e-6))
    if barrier_enabled and model.constraint in {"orthant", "simplex"}:
        barrier_solution = solve_barrier_qp(
            g_full,
            q,
            constraint=model.constraint,
            cap=model.leverage_cap,
            epsilon=barrier_epsilon,
            tolerance=float(config.get("barrier_tolerance", 1e-9)),
            max_iterations=int(
                config.get("barrier_max_iterations", config.get("barrier_max_iter", 120))
            ),
        )
        barrier_consumption = consumption
        if model.has_consumption:
            consumption_barrier = solve_consumption_barrier(
                model,
                estimate.states,
                estimate.lambda_x,
                epsilon=barrier_epsilon,
                tolerance=float(config.get("consumption_barrier_tolerance", 1e-10)),
                max_iterations=int(config.get("consumption_barrier_max_iterations", 120)),
            )
            barrier_consumption = consumption_barrier.consumption
        barrier_control = _combine_control(barrier_solution.control, barrier_consumption)

    reference_risky, reference_consumption = _split_reference(
        model, estimate.reference_control
    )
    full_objective = _joint_objective(
        model, g_full, q, full.control, consumption, estimate.lambda_x
    )
    zero_under_full = _joint_objective(
        model, g_full, q, zero.control, consumption, estimate.lambda_x
    )
    zero_objective = _joint_objective(
        model, g_zero, q, zero.control, consumption, estimate.lambda_x
    )
    reference_objective = _joint_objective(
        model,
        g_full,
        q,
        reference_risky,
        reference_consumption,
        estimate.lambda_x,
    )
    reference_zero_objective = _joint_objective(
        model,
        g_zero,
        q,
        reference_risky,
        reference_consumption,
        estimate.lambda_x,
    )
    shift_linear = g_full - g_zero
    shift_norm = torch.linalg.norm(shift_linear, dim=-1)
    zero_norm = torch.linalg.norm(g_zero, dim=-1)
    zeta_norm = torch.linalg.norm(estimate.zeta_x, dim=-1)
    zeta_se_norm = (
        torch.linalg.norm(estimate.zeta_se[:, 0, :], dim=-1)
        if estimate.zeta_se is not None
        else torch.full_like(zeta_norm, float("nan"))
    )
    zero_full_residual = projected_gradient_residual(
        g_full, q, zero.control, constraint=model.constraint, cap=model.leverage_cap
    )

    diagnostics: dict[str, torch.Tensor] = {
        "zeta_norm": zeta_norm,
        "zeta_se_norm": zeta_se_norm,
        "zeta_signal_to_noise": zeta_norm / zeta_se_norm.clamp_min(1e-14),
        "shift_linear_norm": shift_norm,
        "shift_to_zero_linear_ratio": shift_norm / zero_norm.clamp_min(1e-14),
        "full_zero_action_l2": torch.linalg.norm(full.control - zero.control, dim=-1),
        "full_zero_action_linf": (full.control - zero.control).abs().amax(dim=-1),
        "full_gain_vs_reference": full_objective - reference_objective,
        "zero_gain_vs_reference_on_full_objective": zero_under_full - reference_objective,
        "zero_gain_vs_reference_on_zero_objective": zero_objective - reference_zero_objective,
        "zero_gap_to_full_on_full_objective": full_objective - zero_under_full,
        "reference_kkt_pg": projected_gradient_residual(
            g_full,
            q,
            reference_risky,
            constraint=model.constraint,
            cap=model.leverage_cap,
        ),
        "full_kkt_pg": full.projected_gradient_residual,
        "zero_kkt_pg_for_zero_problem": zero.projected_gradient_residual,
        "zero_kkt_pg_for_full_problem": zero_full_residual,
        "full_qp_converged": full.converged,
        "zero_qp_converged": zero.converged,
        "full_qp_iterations": full.iterations,
        "zero_qp_iterations": zero.iterations,
        "full_qp_fallback": full.used_fallback,
        "zero_qp_fallback": zero.used_fallback,
        "q_regularization": full.regularization,
    }

    if model.has_consumption:
        assert consumption is not None and reference_consumption is not None
        full_consumption_residual = consumption_pg_residual(
            model, estimate.states, estimate.lambda_x, consumption
        )
        reference_consumption_residual = consumption_pg_residual(
            model, estimate.states, estimate.lambda_x, reference_consumption
        )
        lower, upper = model.consumption_bounds(estimate.states)
        width = (upper - lower).clamp_min(1e-14)
        upper_slack = ((upper - consumption) / width).view(-1)
        lower_slack = ((consumption - lower) / width).view(-1)
        switching_band = float(config.get("switching_band", 0.05))
        active_tolerance = float(config.get("active_slack_tolerance", 1e-7))
        # 0 = upper-active, 1 = near upper switch, 2 = far interior.
        region = torch.full_like(upper_slack, 2, dtype=torch.int64)
        region[(upper_slack > active_tolerance) & (upper_slack <= switching_band)] = 1
        region[upper_slack <= active_tolerance] = 0
        diagnostics.update(
            {
                "reference_consumption_kkt_pg": reference_consumption_residual,
                "full_consumption_kkt_pg": full_consumption_residual,
                "zero_consumption_kkt_pg": full_consumption_residual,
                "full_consumption_upper_slack": upper_slack,
                "full_consumption_lower_slack": lower_slack,
                "switching_region_code": region,
                "full_joint_kkt_pg": torch.maximum(
                    full.projected_gradient_residual, full_consumption_residual
                ),
                "reference_joint_kkt_pg": torch.maximum(
                    projected_gradient_residual(
                        g_full,
                        q,
                        reference_risky,
                        constraint=model.constraint,
                        cap=model.leverage_cap,
                    ),
                    reference_consumption_residual,
                ),
            }
        )

    if barrier_solution is not None:
        barrier_consumption = (
            None if not model.has_consumption else barrier_control[:, -1:]
        )
        barrier_objective = _joint_objective(
            model,
            g_full,
            q,
            barrier_solution.control,
            barrier_consumption,
            estimate.lambda_x,
        )
        barrier_portfolio_objective = _portfolio_objective(
            g_full, q, barrier_solution.control
        )
        full_portfolio_objective = _portfolio_objective(g_full, q, full.control)
        diagnostics.update(
            {
                "barrier_kkt_pg": barrier_solution.projected_gradient_residual,
                "barrier_gap_to_full_qp": full_objective - barrier_objective,
                "barrier_portfolio_gap_to_full_qp": (
                    full_portfolio_objective - barrier_portfolio_objective
                ),
                "barrier_gain_vs_reference": barrier_objective - reference_objective,
                "barrier_converged": barrier_solution.converged,
                "barrier_iterations": barrier_solution.iterations,
            }
        )
        if consumption_barrier is not None:
            diagnostics.update(
                {
                    "barrier_consumption_kkt_pg": consumption_pg_residual(
                        model,
                        estimate.states,
                        estimate.lambda_x,
                        consumption_barrier.consumption,
                    ),
                    "barrier_consumption_stationarity": consumption_barrier.residual,
                    "barrier_consumption_converged": consumption_barrier.converged,
                    "barrier_consumption_iterations": consumption_barrier.iterations,
                    "barrier_joint_converged": (
                        barrier_solution.converged & consumption_barrier.converged
                    ),
                }
            )

    diagnostics.update(
        _summary_exact_controls(
            model,
            estimate,
            full_control=full_control,
            zero_control=zero_control,
            barrier_control=barrier_control,
            reference_control=estimate.reference_control,
        )
    )

    if holdout_estimate is not None:
        if holdout_estimate.states.shape != estimate.states.shape or not torch.allclose(
            holdout_estimate.states, estimate.states, rtol=0.0, atol=0.0
        ):
            raise ValueError("Holdout adjoints must use the same diagnostic states")
        if not torch.allclose(holdout_estimate.tau, estimate.tau, rtol=0.0, atol=0.0):
            raise ValueError("Holdout adjoints must use the same diagnostic horizons")
        holdout_g_full, holdout_q = model.local_qp_coefficients(
            holdout_estimate.states,
            holdout_estimate.lambda_,
            holdout_estimate.p_xrow,
            holdout_estimate.zeta_x,
        )
        holdout_g_zero, _ = model.local_qp_coefficients(
            holdout_estimate.states,
            holdout_estimate.lambda_,
            holdout_estimate.p_xrow,
            torch.zeros_like(holdout_estimate.zeta_x),
        )
        holdout_reference_risky, holdout_reference_consumption = _split_reference(
            model, holdout_estimate.reference_control
        )
        full_consumption = None if not model.has_consumption else full_control[:, -1:]
        zero_consumption = None if not model.has_consumption else zero_control[:, -1:]
        holdout_full_value = _joint_objective(
            model,
            holdout_g_full,
            holdout_q,
            full.control,
            full_consumption,
            holdout_estimate.lambda_x,
        )
        holdout_zero_value = _joint_objective(
            model,
            holdout_g_full,
            holdout_q,
            zero.control,
            zero_consumption,
            holdout_estimate.lambda_x,
        )
        holdout_reference_value = _joint_objective(
            model,
            holdout_g_full,
            holdout_q,
            holdout_reference_risky,
            holdout_reference_consumption,
            holdout_estimate.lambda_x,
        )
        holdout_zero_objective_value = _joint_objective(
            model,
            holdout_g_zero,
            holdout_q,
            zero.control,
            zero_consumption,
            holdout_estimate.lambda_x,
        )
        holdout_reference_zero_value = _joint_objective(
            model,
            holdout_g_zero,
            holdout_q,
            holdout_reference_risky,
            holdout_reference_consumption,
            holdout_estimate.lambda_x,
        )
        diagnostics.update(
            {
                "holdout_full_gain_vs_reference": (
                    holdout_full_value - holdout_reference_value
                ),
                "holdout_zero_gain_vs_reference_on_full_objective": (
                    holdout_zero_value - holdout_reference_value
                ),
                "holdout_full_minus_zero_on_full_objective": (
                    holdout_full_value - holdout_zero_value
                ),
                "holdout_zero_gain_vs_reference_on_zero_objective": (
                    holdout_zero_objective_value - holdout_reference_zero_value
                ),
                "zeta_replication_l2": torch.linalg.norm(
                    estimate.zeta_x - holdout_estimate.zeta_x, dim=-1
                ),
                "zeta_holdout_norm": torch.linalg.norm(
                    holdout_estimate.zeta_x, dim=-1
                ),
                "lambda_replication_l2": torch.linalg.norm(
                    estimate.lambda_ - holdout_estimate.lambda_, dim=-1
                ),
                "p_xrow_replication_l2": torch.linalg.norm(
                    estimate.p_xrow - holdout_estimate.p_xrow, dim=-1
                ),
            }
        )
        if barrier_solution is not None and barrier_control is not None:
            holdout_barrier_consumption = (
                None if not model.has_consumption else barrier_control[:, -1:]
            )
            holdout_barrier_value = _joint_objective(
                model,
                holdout_g_full,
                holdout_q,
                barrier_solution.control,
                holdout_barrier_consumption,
                holdout_estimate.lambda_x,
            )
            diagnostics["holdout_barrier_gain_vs_reference"] = (
                holdout_barrier_value - holdout_reference_value
            )
            diagnostics["holdout_full_minus_barrier_on_full_objective"] = (
                holdout_full_value - holdout_barrier_value
            )

    return RecoveryResult(
        full_control=full_control.detach(),
        zero_shift_control=zero_control.detach(),
        barrier_control=None if barrier_control is None else barrier_control.detach(),
        g_full=g_full.detach(),
        g_zero=g_zero.detach(),
        q=q.detach(),
        diagnostics={key: value.detach() for key, value in diagnostics.items()},
        metadata={
            "constraint": model.constraint,
            "leverage_cap": model.leverage_cap,
            "full_shift_default": True,
            "qp_tolerance": tolerance,
            "curvature_floor": curvature_floor,
            "barrier_enabled": barrier_solution is not None,
            "barrier_epsilon": config.get("barrier_epsilon"),
            "holdout_evaluation": holdout_estimate is not None,
            "has_consumption": model.has_consumption,
        },
    )
