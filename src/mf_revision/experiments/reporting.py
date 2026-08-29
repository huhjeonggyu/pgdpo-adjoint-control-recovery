"""Numerical summaries and pointwise CSV exports."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
import torch

from mf_revision.models.base import PortfolioModel
from mf_revision.runtime import write_json
from mf_revision.types import AdjointEstimate, RecoveryResult


def _rmse(estimate: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((estimate - target).square())).cpu())


def _nrmse(estimate: torch.Tensor, target: torch.Tensor) -> float:
    numerator = torch.sqrt(torch.mean((estimate - target).square()))
    denominator = torch.sqrt(torch.mean(target.square())).clamp_min(1e-14)
    return float((numerator / denominator).cpu())


def _mean_relative(estimate: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        torch.mean((estimate - target).abs() / target.abs().clamp_min(1e-14)).cpu()
    )


def _coordinatewise_control_rmse(
    estimate: torch.Tensor, target: torch.Tensor, *, risky_dim: int
) -> float:
    difference = estimate[:, :risky_dim] - target[:, :risky_dim]
    return float(torch.sqrt(torch.mean(difference.square())).cpu())


def adjoint_summary(model: PortfolioModel, estimate: AdjointEstimate) -> dict[str, Any]:
    zeta_norm = torch.linalg.norm(estimate.zeta_x, dim=-1)
    z_norm = torch.linalg.norm(estimate.z_x, dim=-1)
    summary: dict[str, Any] = {
        "mean_zeta_x_norm": float(zeta_norm.mean().cpu()),
        "median_zeta_x_norm": float(zeta_norm.median().cpu()),
        "mean_zeta_to_Z_ratio": float((zeta_norm / z_norm.clamp_min(1e-14)).mean().cpu()),
        "max_zeta_identity_error": float(
            estimate.metadata.get("zeta_identity_max_abs", float("nan"))
        ),
        "graph_mode": estimate.metadata.get("graph_mode"),
        "continuations": estimate.metadata.get("continuations"),
        "independent_mc_units": estimate.metadata.get("independent_mc_units"),
        "requested_z_estimator": estimate.metadata.get("requested_z_estimator"),
        "z_estimator": estimate.metadata.get("z_estimator"),
        "projection_independent_mc_units": estimate.metadata.get(
            "projection_independent_mc_units"
        ),
        "projection_inner_paths": estimate.metadata.get("projection_inner_paths"),
        "projection_total_future_rollouts": estimate.metadata.get(
            "projection_total_future_rollouts"
        ),
        "projection_gram_condition_mean": estimate.metadata.get(
            "projection_gram_condition_mean"
        ),
        "projection_gram_condition_max": estimate.metadata.get(
            "projection_gram_condition_max"
        ),
        "projection_ridge": estimate.metadata.get("projection_ridge"),
        "sample_role": estimate.metadata.get("sample_role"),
        "state_mode": estimate.metadata.get("state_mode"),
    }
    if estimate.lambda_se is not None:
        summary["mean_lambda_se_norm"] = float(
            torch.linalg.norm(estimate.lambda_se, dim=-1).mean().cpu()
        )
    if estimate.p_se is not None:
        summary["mean_P_se_norm"] = float(
            torch.linalg.norm(estimate.p_se.flatten(1), dim=-1).mean().cpu()
        )
    if estimate.z_se is not None:
        summary["mean_Z_se_norm"] = float(
            torch.linalg.norm(estimate.z_se.flatten(1), dim=-1).mean().cpu()
        )
    if estimate.zeta_se is not None:
        zeta_se_norm = torch.linalg.norm(estimate.zeta_se[:, 0, :], dim=-1)
        summary["mean_zeta_se_norm"] = float(zeta_se_norm.mean().cpu())
        summary["mean_zeta_signal_to_noise"] = float(
            (zeta_norm / zeta_se_norm.clamp_min(1e-14)).mean().cpu()
        )

    estimator_names = (
        "raw_moment",
        "centered_moment",
        "control_variate",
        "ols_control_variate",
    )
    for estimator_name in estimator_names:
        zeta_key = f"{estimator_name}_zeta"
        z_key = f"{estimator_name}_z"
        if zeta_key not in estimate.projection_variants:
            continue
        variant_zeta = estimate.projection_variants[zeta_key][:, 0, :]
        variant_norm = torch.linalg.norm(variant_zeta, dim=-1)
        summary[f"{estimator_name}_mean_zeta_x_norm"] = float(
            variant_norm.mean().cpu()
        )
        se_tensor = estimate.projection_se_variants.get(zeta_key)
        if se_tensor is not None:
            variant_se_norm = torch.linalg.norm(se_tensor[:, 0, :], dim=-1)
            summary[f"{estimator_name}_mean_zeta_se_norm"] = float(
                variant_se_norm.mean().cpu()
            )
            summary[f"{estimator_name}_mean_zeta_signal_to_noise"] = float(
                (variant_norm / variant_se_norm.clamp_min(1e-14)).mean().cpu()
            )
        if z_key in estimate.projection_variants:
            summary[f"{estimator_name}_mean_Z_x_norm"] = float(
                torch.linalg.norm(estimate.projection_variants[z_key][:, 0, :], dim=-1)
                .mean()
                .cpu()
            )

    try:
        exact = model.exact_adjoint_fields(estimate.states, estimate.tau)
    except NotImplementedError:
        return summary

    if "lambda" in exact:
        summary["lambda_x_nrmse"] = _nrmse(
            estimate.lambda_x, exact["lambda"][:, 0:1]
        )
        summary["lambda_x_mean_relative_error"] = _mean_relative(
            estimate.lambda_x, exact["lambda"][:, 0:1]
        )
    if "P_wealth_row" in exact:
        summary["P_xrow_nrmse"] = _nrmse(
            estimate.p_xrow, exact["P_wealth_row"]
        )
    if "PXX" in exact:
        summary["PXX_nrmse"] = _nrmse(estimate.p[:, 0, 0:1], exact["PXX"])
    if "VXY" in exact and model.dims.factor:
        # Diagnostic discrepancy only; VXY is not labeled as the PMP PXY target.
        summary["PXY_vs_VXY_nrmse"] = _nrmse(
            estimate.p_xrow[:, 1 : 1 + model.dims.factor], exact["VXY"]
        )
    if "Z_wealth" in exact:
        summary["Z_x_nrmse"] = _nrmse(estimate.z_x, exact["Z_wealth"])
    if "zeta_wealth" in exact:
        summary["zeta_x_rmse_to_exact"] = _rmse(
            estimate.zeta_x, exact["zeta_wealth"]
        )
    if "control" in exact:
        summary["reference_control_rmse_to_exact"] = _rmse(
            estimate.reference_control, exact["control"]
        )
        summary["reference_risky_coordinate_rmse_to_exact"] = (
            _coordinatewise_control_rmse(
                estimate.reference_control,
                exact["control"],
                risky_dim=model.dims.risky,
            )
        )

    for estimator_name in estimator_names:
        z_key = f"{estimator_name}_z"
        zeta_key = f"{estimator_name}_zeta"
        if "Z_wealth" in exact and z_key in estimate.projection_variants:
            summary[f"{estimator_name}_Z_x_nrmse"] = _nrmse(
                estimate.projection_variants[z_key][:, 0, :], exact["Z_wealth"]
            )
        if "zeta_wealth" in exact and zeta_key in estimate.projection_variants:
            summary[f"{estimator_name}_zeta_x_rmse_to_exact"] = _rmse(
                estimate.projection_variants[zeta_key][:, 0, :],
                exact["zeta_wealth"],
            )
    return summary


def recovery_summary(result: RecoveryResult) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, tensor in result.diagnostics.items():
        value = tensor.detach().cpu()
        if value.dtype == torch.bool:
            output[f"{key}_fraction"] = float(value.double().mean())
            continue
        flat = value.double().reshape(-1)
        finite = flat[torch.isfinite(flat)]
        if finite.numel() == 0:
            output[f"{key}_mean"] = float("nan")
            continue
        output[f"{key}_mean"] = float(finite.mean())
        output[f"{key}_median"] = float(finite.median())
        output[f"{key}_max"] = float(finite.max())
        output[f"{key}_min"] = float(finite.min())
    return output


def write_pointwise(
    path: str | Path,
    estimate: AdjointEstimate,
    recovery: RecoveryResult | None = None,
) -> None:
    data: dict[str, Any] = {}
    state = estimate.states.detach().cpu()
    for index in range(state.shape[1]):
        data["X" if index == 0 else f"Y{index}"] = state[:, index].numpy()
    data["tau"] = estimate.tau[:, 0].detach().cpu().numpy()
    for index in range(estimate.lambda_.shape[1]):
        data[f"lambda_{index}"] = estimate.lambda_[:, index].detach().cpu().numpy()
    for index in range(estimate.p_xrow.shape[1]):
        data[f"P_X{index}"] = estimate.p_xrow[:, index].detach().cpu().numpy()
    for index in range(estimate.z_x.shape[1]):
        data[f"Z_X_{index}"] = estimate.z_x[:, index].detach().cpu().numpy()
        data[f"zeta_X_{index}"] = estimate.zeta_x[:, index].detach().cpu().numpy()
        if estimate.zeta_se is not None:
            data[f"zeta_X_se_{index}"] = (
                estimate.zeta_se[:, 0, index].detach().cpu().numpy()
            )
    for estimator_name in (
        "raw_moment",
        "centered_moment",
        "control_variate",
        "ols_control_variate",
    ):
        z_key = f"{estimator_name}_z"
        zeta_key = f"{estimator_name}_zeta"
        if z_key not in estimate.projection_variants:
            continue
        variant_z = estimate.projection_variants[z_key][:, 0, :]
        variant_zeta = estimate.projection_variants[zeta_key][:, 0, :]
        variant_se = estimate.projection_se_variants.get(zeta_key)
        for index in range(variant_z.shape[1]):
            data[f"Z_X_{index}_{estimator_name}"] = variant_z[:, index].detach().cpu().numpy()
            data[f"zeta_X_{index}_{estimator_name}"] = (
                variant_zeta[:, index].detach().cpu().numpy()
            )
            if variant_se is not None:
                data[f"zeta_X_se_{index}_{estimator_name}"] = (
                    variant_se[:, 0, index].detach().cpu().numpy()
                )

    if recovery is not None:
        for index in range(recovery.full_control.shape[1]):
            data[f"full_control_{index}"] = recovery.full_control[:, index].cpu().numpy()
            data[f"zero_control_{index}"] = recovery.zero_shift_control[:, index].cpu().numpy()
            if recovery.barrier_control is not None:
                data[f"barrier_control_{index}"] = recovery.barrier_control[:, index].cpu().numpy()
        for key, value in recovery.diagnostics.items():
            if value.ndim == 1:
                data[key] = value.cpu().numpy()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(target, index=False)


def write_summaries(
    output_dir: str | Path,
    model: PortfolioModel,
    estimate: AdjointEstimate,
    recovery: RecoveryResult | None = None,
) -> None:
    output = Path(output_dir)
    write_json(output / "adjoint_summary.json", adjoint_summary(model, estimate))
    if recovery is not None:
        write_json(output / "recovery_summary.json", recovery_summary(recovery))
    write_pointwise(output / "pointwise.csv", estimate, recovery)
