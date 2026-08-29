"""Variance-reduced one-step Brownian projections for ``Z`` and ``zeta``."""
from __future__ import annotations

from dataclasses import dataclass
import torch


_METHODS = {
    "raw_moment",
    "centered_moment",
    "control_variate",
    "ols_control_variate",
}


@dataclass(slots=True)
class ProjectionResult:
    z: torch.Tensor
    zeta: torch.Tensor
    z_se: torch.Tensor
    zeta_se: torch.Tensor
    units: int
    diagnostics: dict[str, float | str]


def paired_projection_units(
    response: torch.Tensor,
    delta_w: torch.Tensor,
    *,
    paired: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return independent regression units.

    Consecutive paired paths are replaced by half differences.  When only the
    first Brownian innovation is sign-flipped and all future innovations are
    shared, this is a common-random-number central difference for the adapted
    next-step costate.
    """
    if not paired:
        return response, delta_w
    if response.shape[0] % 2:
        raise ValueError("Paired projection batches require complete adjacent pairs")
    response_pair = response.reshape(response.shape[0] // 2, 2, *response.shape[1:])
    brownian_pair = delta_w.reshape(delta_w.shape[0] // 2, 2, delta_w.shape[1])
    return (
        0.5 * (response_pair[:, 0] - response_pair[:, 1]),
        0.5 * (brownian_pair[:, 0] - brownian_pair[:, 1]),
    )


def _sample_standard_error(samples: torch.Tensor) -> torch.Tensor:
    if samples.shape[0] <= 1:
        return torch.full_like(samples[0], float("nan"))
    return samples.std(dim=0, unbiased=True) / float(samples.shape[0]) ** 0.5


def _ols_slope_and_se(
    response: torch.Tensor,
    design: torch.Tensor,
    *,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """OLS slope with intercept and an HC1 componentwise standard error."""
    if response.ndim != 2 or design.ndim != 2 or response.shape[0] != design.shape[0]:
        raise ValueError("Expected response[N,n] and design[N,q]")
    count, brownian_dim = design.shape
    if count <= brownian_dim + 1:
        raise ValueError(
            "OLS Brownian projection requires more independent units than "
            f"Brownian dimensions plus an intercept; got N={count}, q={brownian_dim}"
        )
    centered_response = response - response.mean(dim=0, keepdim=True)
    centered_design = design - design.mean(dim=0, keepdim=True)
    gram = centered_design.transpose(0, 1) @ centered_design
    trace_scale = torch.trace(gram) / float(max(1, brownian_dim))
    regularization = float(ridge) * trace_scale.clamp_min(torch.finfo(gram.dtype).eps)
    gram_regularized = gram + regularization * torch.eye(
        brownian_dim, device=gram.device, dtype=gram.dtype
    )
    cross = centered_design.transpose(0, 1) @ centered_response
    beta_qn = torch.linalg.solve(gram_regularized, cross)
    slope = beta_qn.transpose(0, 1)

    fitted = centered_design @ beta_qn
    residual = centered_response - fitted
    gram_inverse = torch.linalg.inv(gram_regularized)
    degrees = max(1, count - brownian_dim - 1)
    hc1 = float(count) / float(degrees)
    standard_error = torch.empty_like(slope)
    for output_index in range(response.shape[1]):
        weighted_design = centered_design * residual[:, output_index : output_index + 1]
        meat = weighted_design.transpose(0, 1) @ weighted_design
        covariance = hc1 * (gram_inverse @ meat @ gram_inverse)
        standard_error[output_index] = torch.diag(covariance).clamp_min(0.0).sqrt()
    condition = float(torch.linalg.cond(gram_regularized).detach().cpu())
    return slope, standard_error, condition


def estimate_brownian_projection(
    response_units: torch.Tensor,
    delta_w_units: torch.Tensor,
    *,
    dt: float,
    p_sigma: torch.Tensor,
    p_sigma_se: torch.Tensor,
    method: str = "ols_control_variate",
    ridge: float = 1.0e-12,
) -> ProjectionResult:
    """Estimate ``Z`` and ``zeta=Z-P sigma_ref``.

    The control-variate estimators work with

    ``lambda_{t+dt} - (P sigma_ref) DeltaW``

    before taking the Brownian projection.  The OLS version additionally uses
    the realized Brownian Gram matrix rather than replacing it by ``dt I``.
    The main ``P`` bank and the projection bank are independent by default, so
    their standard errors are combined in quadrature.
    """
    method = str(method).lower().replace("-", "_")
    aliases = {
        "raw": "raw_moment",
        "centered": "centered_moment",
        "cv": "control_variate",
        "ols": "ols_control_variate",
    }
    method = aliases.get(method, method)
    if method not in _METHODS:
        raise ValueError(f"projection method must be one of {sorted(_METHODS)}")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if response_units.ndim != 2 or delta_w_units.ndim != 2:
        raise ValueError("Expected response_units[N,n] and delta_w_units[N,q]")
    if response_units.shape[0] != delta_w_units.shape[0]:
        raise ValueError("Projection response/design count mismatch")
    if p_sigma.shape != (response_units.shape[1], delta_w_units.shape[1]):
        raise ValueError("p_sigma shape mismatch")

    count = response_units.shape[0]
    condition = float("nan")
    if method == "raw_moment":
        z_samples = response_units.unsqueeze(-1) * delta_w_units.unsqueeze(1) / float(dt)
        z = z_samples.mean(dim=0)
        z_se = _sample_standard_error(z_samples)
        zeta = z - p_sigma
        zeta_se = torch.sqrt(z_se.square() + p_sigma_se.square())
    elif method == "centered_moment":
        response = response_units - response_units.mean(dim=0, keepdim=True)
        design = delta_w_units - delta_w_units.mean(dim=0, keepdim=True)
        z_samples = response.unsqueeze(-1) * design.unsqueeze(1) / float(dt)
        z = z_samples.mean(dim=0)
        z_se = _sample_standard_error(z_samples)
        zeta = z - p_sigma
        zeta_se = torch.sqrt(z_se.square() + p_sigma_se.square())
        gram = design.transpose(0, 1) @ design
        condition = float(torch.linalg.cond(gram).detach().cpu())
    elif method == "control_variate":
        response = response_units - response_units.mean(dim=0, keepdim=True)
        design = delta_w_units - delta_w_units.mean(dim=0, keepdim=True)
        residual = response - design @ p_sigma.transpose(0, 1)
        zeta_samples = residual.unsqueeze(-1) * design.unsqueeze(1) / float(dt)
        zeta = zeta_samples.mean(dim=0)
        zeta_regression_se = _sample_standard_error(zeta_samples)
        z = p_sigma + zeta
        zeta_se = torch.sqrt(zeta_regression_se.square() + p_sigma_se.square())
        z_se = zeta_se
        gram = design.transpose(0, 1) @ design
        condition = float(torch.linalg.cond(gram).detach().cpu())
    else:
        centered_response = response_units - response_units.mean(dim=0, keepdim=True)
        centered_design = delta_w_units - delta_w_units.mean(dim=0, keepdim=True)
        residual = centered_response - centered_design @ p_sigma.transpose(0, 1)
        zeta, zeta_regression_se, condition = _ols_slope_and_se(
            residual, centered_design, ridge=float(ridge)
        )
        z = p_sigma + zeta
        zeta_se = torch.sqrt(zeta_regression_se.square() + p_sigma_se.square())
        z_se = zeta_se

    identity_error = float((zeta - (z - p_sigma)).abs().max().detach().cpu())
    return ProjectionResult(
        z=z,
        zeta=zeta,
        z_se=z_se,
        zeta_se=zeta_se,
        units=int(count),
        diagnostics={
            "projection_method": method,
            "projection_gram_condition": condition,
            "zeta_identity_error": identity_error,
            "ridge": float(ridge),
        },
    )
