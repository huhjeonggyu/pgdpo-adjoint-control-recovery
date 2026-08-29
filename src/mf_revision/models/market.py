"""Deterministic market generators matching the paper's legacy calibrations."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import numpy as np
import torch


def nearest_spd_correlation(correlation: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    correlation = 0.5 * (correlation + correlation.T)
    values, vectors = torch.linalg.eigh(correlation)
    values = values.clamp_min(eps)
    correlation = vectors @ torch.diag(values) @ vectors.T
    scale = torch.diag(correlation).clamp_min(eps).sqrt()
    inverse = torch.diag(1.0 / scale)
    normalized = inverse @ correlation @ inverse
    return 0.5 * (normalized + normalized.T)




def _ridge_asset_loading_preserve_factor_cross(
    asset_loading: torch.Tensor,
    factor_loading: torch.Tensor,
    ridge: float | torch.Tensor,
) -> torch.Tensor:
    """Add isotropic asset covariance ridge while preserving factor cross-covariance.

    The Brownian basis and factor loading are kept fixed.  Only the component of
    the asset loading orthogonal to the factor-loading row space is recolored, so

        A_new A_new^T = A A^T + ridge I,
        A_new F^T     = A F^T.

    For the paper generators q=d+k and rank(F)=k, the orthogonal component has
    dimension d and its covariance is positive definite.  The computation is
    carried out in float64 for numerical stability and cast back afterwards.
    """
    if asset_loading.ndim != 2 or factor_loading.ndim != 2:
        raise ValueError("asset_loading and factor_loading must be matrices")
    d, q = asset_loading.shape
    k, q_factor = factor_loading.shape
    if q_factor != q:
        raise ValueError("asset and factor loadings must use the same Brownian basis")
    ridge_value = float(ridge)
    if ridge_value < 0.0:
        raise ValueError("ridge must be nonnegative")
    if ridge_value == 0.0:
        return asset_loading

    work_dtype = torch.float64
    A = asset_loading.to(dtype=work_dtype)
    F = factor_loading.to(dtype=work_dtype)
    eye_d = torch.eye(d, device=A.device, dtype=work_dtype)

    if k == 0:
        target = A @ A.T + ridge_value * eye_d
        return torch.linalg.cholesky(target).to(dtype=asset_loading.dtype)

    gram = F @ F.T
    projection = F.T @ torch.linalg.solve(gram, F)
    A_parallel = A @ projection
    A_perp = A - A_parallel

    residual = 0.5 * (A_perp @ A_perp.T + (A_perp @ A_perp.T).T)
    target_residual = residual + ridge_value * eye_d
    chol_old = torch.linalg.cholesky(residual)
    chol_new = torch.linalg.cholesky(target_residual)
    whitened = torch.linalg.solve_triangular(chol_old, A_perp, upper=False)
    A_perp_new = chol_new @ whitened
    A_new = A_parallel + A_perp_new
    return A_new.to(dtype=asset_loading.dtype)

def load_market_snapshot(
    path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | Any]:
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid market snapshot: {path}")
    output: dict[str, torch.Tensor | Any] = {}
    for key, value in payload.items():
        output[key] = value.to(device=device, dtype=dtype) if torch.is_tensor(value) else value
    return output


def legacy_merton_short_market(
    d: int,
    *,
    gamma: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    vol_range: tuple[float, float] = (0.18, 0.28),
    average_correlation: float = 0.50,
    correlation_jitter: float = 0.06,
    alpha_range: tuple[float, float] = (0.05, 0.22),
    short_fraction: float = 0.25,
    hedge_alpha_range: tuple[float, float] = (-0.03, 0.02),
    enforce_shorts: bool = True,
    target_negative: int = 1,
    max_tries: int = 6,
    target_sum: float | None = None,
    sum_range: tuple[float, float] = (0.40, 0.75),
    sum_bias_exp: float = 0.7,
    rho_div: float = 0.35,
    wmax_cap: float = 0.70,
    short_mass_frac: float = 0.15,
    short_count_max_frac: float = 0.40,
    lambda_factor: float = 2e-3,
    generation_dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Exact port of the final historical ``mt_nd_short`` market builder."""
    devices: list[int] = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))

        def sample_covariance(avg_corr: float, jitter: float) -> torch.Tensor:
            sigma = torch.empty(d, device=device, dtype=generation_dtype).uniform_(
                *vol_range
            )
            psi = torch.full(
                (d, d), float(avg_corr), device=device, dtype=generation_dtype
            )
            psi.fill_diagonal_(1.0)
            if jitter > 0:
                noise = torch.randn(d, d, device=device, dtype=generation_dtype) * jitter
                psi = nearest_spd_correlation(psi + 0.5 * (noise + noise.T))
            covariance = torch.diag(sigma) @ psi @ torch.diag(sigma)
            ridge = float(lambda_factor) * float(covariance.diag().mean())
            return covariance + ridge * torch.eye(
                d, device=device, dtype=generation_dtype
            )

        corr, jitter = float(average_correlation), float(correlation_jitter)
        for _ in range(int(max_tries)):
            covariance = sample_covariance(corr, jitter)
            covariance_inv = torch.linalg.inv(covariance)
            alpha = torch.empty(d, device=device, dtype=generation_dtype).uniform_(
                *alpha_range
            )
            if short_fraction > 0:
                count = max(1, int(d * short_fraction))
                index = torch.randperm(d, device=device)[:count]
                alpha[index] = torch.empty(
                    count, device=device, dtype=generation_dtype
                ).uniform_(*hedge_alpha_range)
            unconstrained = covariance_inv @ alpha / float(gamma)
            if (not enforce_shorts) or int((unconstrained < 0).sum()) >= int(target_negative):
                break
            corr = min(0.80, corr + 0.08)
            jitter = min(0.12, jitter + 0.02)

        if target_sum is None:
            low, high = map(float, sum_range)
            random = torch.rand((), device=device, dtype=generation_dtype)
            selected_sum = float((low + (high - low) * random.pow(sum_bias_exp)).item())
        else:
            selected_sum = float(target_sum)
        one = torch.ones(d, device=device, dtype=generation_dtype)
        alpha = alpha + float(gamma) * (
            selected_sum - float(unconstrained.sum())
        ) * (covariance @ one) / float(d)
        control = covariance_inv @ alpha / float(gamma)

        def project_simplex(vector: torch.Tensor, mass: float) -> torch.Tensor:
            positive = vector.clamp_min(0.0)
            if float(positive.sum()) <= 1e-12:
                return torch.full_like(vector, mass / float(d))
            ordered, _ = torch.sort(positive, descending=True)
            cumulative = torch.cumsum(ordered, 0) - mass
            index = torch.arange(1, d + 1, device=device, dtype=vector.dtype)
            condition = ordered > cumulative / index
            if bool(condition.any()):
                rho = int(torch.nonzero(condition, as_tuple=False)[-1].item())
                threshold = cumulative[rho] / float(rho + 1)
            else:
                threshold = cumulative[-1] / float(d)
            return (positive - threshold).clamp_min(0.0)

        selected_sum = float(control.sum())
        positive_control = project_simplex(control, selected_sum)
        control = (1.0 - float(rho_div)) * control + float(rho_div) * positive_control

        negative = (-control).clamp_min(0.0)
        negative_mass = float(negative.sum())
        mass_cap = float(short_mass_frac) * selected_sum
        if negative_mass > mass_cap + 1e-12:
            scale = mass_cap / (negative_mass + 1e-12)
            delta = (1.0 - scale) * negative
            control[control < 0] = -scale * negative[control < 0]
            positive_index = control > 0
            positive_mass = float(control[positive_index].sum())
            if positive_mass > 1e-12:
                control[positive_index] -= control[positive_index] * (
                    delta.sum() / positive_mass
                )

        maximum_negative_count = int(float(short_count_max_frac) * d)
        negative_index = (control < 0).nonzero(as_tuple=False).view(-1)
        if negative_index.numel() > maximum_negative_count:
            values = control[negative_index].abs()
            _, order = torch.sort(values)
            kill = negative_index[
                order[: negative_index.numel() - maximum_negative_count]
            ]
            freed = (-control[kill]).sum()
            control[kill] = 0.0
            positive_index = control > 0
            positive_mass = float(control[positive_index].sum())
            if positive_mass > 1e-12:
                control[positive_index] -= control[positive_index] * (
                    freed / positive_mass
                )

        if wmax_cap is not None:
            over = control > float(wmax_cap)
            if bool(over.any()):
                excess = (control[over] - float(wmax_cap)).sum()
                control[over] = float(wmax_cap)
                positive_index = (control > 0) & (~over)
                positive_mass = float(control[positive_index].sum())
                if positive_mass > 1e-12:
                    control[positive_index] -= control[positive_index] * (
                        excess / positive_mass
                    )
                else:
                    control += excess / float(d)

        alpha = float(gamma) * (covariance @ control)
        covariance_inv = torch.linalg.inv(covariance)

    return {
        "alpha": alpha.to(device=device, dtype=dtype),
        "covariance": covariance.to(device=device, dtype=dtype),
        "covariance_inv": covariance_inv.to(device=device, dtype=dtype),
        "unconstrained_reference": control.to(device=device, dtype=dtype),
    }

def legacy_merton_cap_market(
    d: int,
    *,
    gamma: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    generation_dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Exact k=0 port of the Stage-5b Merton consumption-cap market."""
    rng = np.random.default_rng(int(seed))
    sigma = torch.linspace(0.18, 0.30, d, device=device, dtype=generation_dtype)
    psi = torch.full((d, d), 0.25, device=device, dtype=generation_dtype)
    psi.fill_diagonal_(1.0)
    noise = torch.tensor(
        rng.normal(0.0, 0.035, size=(d, d)),
        device=device,
        dtype=generation_dtype,
    )
    psi = nearest_spd_correlation(psi + 0.5 * (noise + noise.T), eps=1e-5)
    base_covariance = torch.diag(sigma) @ psi @ torch.diag(sigma)
    ridge = 1e-4 * base_covariance.diag().mean()
    covariance = base_covariance + ridge * torch.eye(
        d, device=device, dtype=generation_dtype
    )
    # Use a Brownian loading consistent with the ridged covariance.
    diffusion_loading = torch.linalg.cholesky(covariance)
    alpha = torch.linspace(0.015, 0.075, d, device=device, dtype=generation_dtype)
    alpha_out = alpha.to(device=device, dtype=dtype)
    loading_out = diffusion_loading.to(device=device, dtype=dtype)
    covariance_out = loading_out @ loading_out.T
    unconstrained_reference = torch.linalg.solve(covariance_out, alpha_out) / float(gamma)
    return {
        "alpha": alpha_out,
        "covariance": covariance_out,
        "covariance_inv": torch.linalg.inv(covariance_out),
        "unconstrained_reference": unconstrained_reference,
        "sigma": sigma.to(device=device, dtype=dtype),
        "loading": loading_out,
    }

def _corr_from_latent_beta(beta: torch.Tensor) -> torch.Tensor:
    correlation = torch.outer(beta, beta)
    correlation.fill_diagonal_(1.0)
    return nearest_spd_correlation(correlation)


def _random_corr(k: int, rng: np.random.Generator, dtype: torch.dtype) -> torch.Tensor:
    if k <= 1:
        return torch.eye(k, dtype=dtype)
    matrix = rng.normal(size=(k, k))
    covariance = matrix @ matrix.T
    scale = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    correlation = covariance / np.outer(scale, scale)
    return nearest_spd_correlation(torch.tensor(correlation, dtype=dtype))


def affine_joint_market(
    d: int,
    k: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    mode: str = "old_affine",
) -> dict[str, torch.Tensor]:
    """Port of the final affine-factor market builders."""
    rng = np.random.default_rng(int(seed))
    generation_dtype = torch.float32
    if mode not in {"old_affine", "conservative"}:
        raise ValueError("mode must be old_affine or conservative")

    if mode == "old_affine":
        if k <= 0:
            raise ValueError("old_affine requires at least one factor")
        if k == 1:
            kappa = torch.tensor([2.0], dtype=generation_dtype)
        else:
            kappa = torch.tensor(
                rng.uniform(2.0, 2.0 + 0.5 * (k - 1), size=k), dtype=generation_dtype
            )
        theta = torch.tensor(rng.uniform(0.2, 0.4, size=k), dtype=generation_dtype)
        factor_volatility = torch.tensor(
            rng.uniform(0.3, 0.5, size=k), dtype=generation_dtype
        )
        volatility = torch.tensor(rng.uniform(0.2, 0.4, size=d), dtype=generation_dtype)
        beta = torch.tensor(rng.uniform(-0.8, 0.8, size=d), dtype=generation_dtype)
        asset_correlation = _corr_from_latent_beta(beta)
        factor_correlation = _random_corr(k, rng, generation_dtype)
        cross = torch.tensor(rng.uniform(-0.2, 0.2, size=(d, k)), dtype=generation_dtype)
        raw = rng.dirichlet(np.ones(k), size=d) if k > 1 else np.ones((d, 1))
        alpha_load = volatility.view(-1, 1) * torch.tensor(raw, dtype=generation_dtype)
        alpha0 = torch.zeros(d, dtype=generation_dtype)
    else:
        volatility = torch.linspace(0.18, 0.30, d, dtype=generation_dtype)
        asset_correlation = torch.full((d, d), 0.25, dtype=generation_dtype)
        asset_correlation.fill_diagonal_(1.0)
        noise = torch.tensor(rng.normal(0.0, 0.035, size=(d, d)), dtype=generation_dtype)
        asset_correlation = nearest_spd_correlation(
            asset_correlation + 0.5 * (noise + noise.T)
        )
        kappa = torch.linspace(0.7, 1.4, k, dtype=generation_dtype)
        theta = torch.zeros(k, dtype=generation_dtype)
        factor_volatility = torch.linspace(0.12, 0.20, k, dtype=generation_dtype)
        factor_correlation = torch.eye(k, dtype=generation_dtype)
        cross = torch.tensor(
            rng.normal(0.0, 0.20 / math.sqrt(max(k, 1)), size=(d, k)),
            dtype=generation_dtype,
        )
        alpha_load = torch.zeros(d, k, dtype=generation_dtype)
        alpha_load[:, 0] = (
            torch.where(torch.arange(d) % 2 == 0, 1.0, -1.0)
            * torch.linspace(0.05, 0.22, d)
        )
        if k > 1:
            alpha_load[:, 1:] = torch.tensor(
                rng.normal(0.0, 0.10 / math.sqrt(k - 1), size=(d, k - 1)),
                dtype=generation_dtype,
            )
        alpha0 = torch.linspace(0.015, 0.075, d, dtype=generation_dtype)

    block = torch.zeros(d + k, d + k, dtype=generation_dtype)
    block[:d, :d] = asset_correlation
    block[d:, d:] = factor_correlation
    block[:d, d:] = cross
    block[d:, :d] = cross.T
    block = nearest_spd_correlation(block)
    joint_cholesky = torch.linalg.cholesky(block)
    asset_loading = torch.diag(volatility) @ joint_cholesky[:d, :]
    factor_loading = torch.diag(factor_volatility) @ joint_cholesky[d:, :]
    base_covariance = asset_loading @ asset_loading.T
    ridge = 1e-4 * base_covariance.diag().mean()
    covariance = base_covariance + ridge * torch.eye(d, dtype=generation_dtype)
    # Recolor only the factor-orthogonal asset-noise component.  This preserves
    # the common Brownian basis and asset/factor cross covariance while making
    # the simulator loading exactly consistent with the ridged asset covariance.
    asset_loading = _ridge_asset_loading_preserve_factor_cross(
        asset_loading, factor_loading, ridge
    )
    covariance = asset_loading @ asset_loading.T
    cross_covariance = asset_loading @ factor_loading.T
    asset_loading_out = asset_loading.to(device=device, dtype=dtype)
    factor_loading_out = factor_loading.to(device=device, dtype=dtype)
    covariance_out = asset_loading_out @ asset_loading_out.T
    cross_covariance_out = asset_loading_out @ factor_loading_out.T
    return {
        "alpha0": alpha0.to(device=device, dtype=dtype),
        "alpha_load": alpha_load.to(device=device, dtype=dtype),
        "kappa": kappa.to(device=device, dtype=dtype),
        "theta": theta.to(device=device, dtype=dtype),
        "asset_loading": asset_loading_out,
        "factor_loading": factor_loading_out,
        "covariance": covariance_out,
        "cross_covariance": cross_covariance_out,
    }


# Backward-compatible name used by the initial clean revalidation release.
def legacy_merton_market(
    d: int,
    *,
    gamma: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    market = legacy_merton_short_market(
        d,
        gamma=gamma,
        seed=seed,
        device=device,
        dtype=dtype,
        **kwargs,
    )
    return market["alpha"], market["covariance"]
