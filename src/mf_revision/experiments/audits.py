"""Small numerical audits that reuse harvested paper adjoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd
import torch

from mf_revision.models.base import PortfolioModel
from mf_revision.recovery.qp import projected_gradient_residual
from mf_revision.types import AdjointEstimate, RecoveryResult


def _project_simplex(values: torch.Tensor, cap: float) -> torch.Tensor:
    # Batched Euclidean projection onto {u>=0, sum u<=cap}.
    rows: list[torch.Tensor] = []
    for row in values:
        positive = row.clamp_min(0.0)
        if float(positive.sum()) <= float(cap):
            rows.append(positive)
            continue
        ordered, _ = torch.sort(positive, descending=True)
        cumulative = torch.cumsum(ordered, dim=0) - float(cap)
        index = torch.arange(1, row.numel() + 1, device=row.device, dtype=row.dtype)
        valid = ordered - cumulative / index > 0
        rho = int(torch.nonzero(valid, as_tuple=False)[-1])
        threshold = cumulative[rho] / float(rho + 1)
        rows.append((positive - threshold).clamp_min(0.0))
    return torch.stack(rows)


def _project(model: PortfolioModel, values: torch.Tensor) -> torch.Tensor:
    if model.constraint == "unconstrained":
        return values
    if model.constraint == "orthant":
        return values.clamp_min(0.0)
    if model.constraint == "simplex":
        return _project_simplex(values, model.leverage_cap)
    raise ValueError(f"Unknown constraint: {model.constraint}")


def _pgd(
    model: PortfolioModel,
    g: torch.Tensor,
    q: torch.Tensor,
    start: torch.Tensor,
    *,
    iterations: int,
) -> torch.Tensor:
    eigmax = torch.linalg.eigvalsh(0.5 * (q + q.transpose(1, 2))).amax(dim=-1)
    step = (0.99 / eigmax.clamp_min(1e-12)).view(-1, 1)
    control = _project(model, start)
    for _ in range(int(iterations)):
        gradient = g - torch.einsum("bij,bj->bi", q, control)
        control = _project(model, control + step * gradient)
    return control


def run_initialization_audit(
    model: PortfolioModel,
    adjoint_path: str | Path,
    recovery_path: str | Path,
    output_dir: str | Path,
    *,
    states: int | None = None,
    iterations: int = 2000,
    seed: int = 12345,
) -> dict[str, Any]:
    estimate = AdjointEstimate.load(adjoint_path, map_location=model.device)  # type: ignore[attr-defined]
    recovery = RecoveryResult.load(recovery_path, map_location=model.device)  # type: ignore[attr-defined]
    count = estimate.states.shape[0] if states is None else min(int(states), estimate.states.shape[0])
    g, q = recovery.g_full[:count], recovery.q[:count]
    target = recovery.full_control[:count, : model.dims.risky]
    reference = estimate.reference_control[:count, : model.dims.risky]
    dimension = model.dims.risky
    if model.constraint == "simplex":
        equal = torch.full_like(reference, model.leverage_cap / float(dimension + 1))
    else:
        equal = torch.full_like(reference, 1.0 / float(max(dimension, 1)))
    zero = torch.zeros_like(reference)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random = torch.rand(reference.shape, generator=generator, dtype=reference.dtype).to(reference)
    random = _project(model, random)
    starts = {
        "DPO": reference,
        "equal": equal,
        "zero": zero,
        "random": random,
    }
    rows: list[dict[str, Any]] = []
    all_differences: list[torch.Tensor] = []
    for label, start in starts.items():
        result = _pgd(model, g, q, start, iterations=iterations)
        difference = torch.linalg.norm(result - target, dim=-1)
        all_differences.append(difference)
        residual = projected_gradient_residual(
            g,
            q,
            result,
            constraint=model.constraint,
            cap=model.leverage_cap,
        )
        for index in range(count):
            rows.append(
                {
                    "state": index,
                    "initialization": label,
                    "l2_to_exact_qp": float(difference[index].cpu()),
                    "kkt_pg": float(residual[index].cpu()),
                }
            )
    stacked = torch.stack(all_differences)
    summary = {
        "states": count,
        "iterations": int(iterations),
        "initializations": list(starts),
        "mean_difference": float(stacked.mean().cpu()),
        "maximum_difference": float(stacked.max().cpu()),
        "exact_qp_uses_initialization": False,
    }
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "initialization_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(target_dir / "initialization_audit_pointwise.csv", index=False)
    return summary


def compare_graph_estimates(
    ol_path: str | Path,
    cl_path: str | Path,
    output_path: str | Path,
) -> dict[str, float]:
    ol = AdjointEstimate.load(ol_path, map_location="cpu")
    cl = AdjointEstimate.load(cl_path, map_location="cpu")
    if not torch.equal(ol.states, cl.states) or not torch.equal(ol.tau, cl.tau):
        raise ValueError("OL and CL graph audits must use identical diagnostic states")

    def nrmse(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(
            (
                torch.sqrt(torch.mean((a - b).square()))
                / torch.sqrt(torch.mean(a.square())).clamp_min(1e-14)
            ).cpu()
        )

    values = {
        "lambda_X_dnRMSE": nrmse(ol.lambda_x, cl.lambda_x),
        "PXX_dnRMSE": nrmse(ol.p[:, 0, 0], cl.p[:, 0, 0]),
        "PXY_dnRMSE": (
            nrmse(ol.p_xrow[:, 1:], cl.p_xrow[:, 1:])
            if ol.p_xrow.shape[1] > 1
            else float("nan")
        ),
        "Z_X_dnRMSE": nrmse(ol.z_x, cl.z_x),
        "zeta_X_dnRMSE": nrmse(ol.zeta_x, cl.zeta_x),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(values, indent=2), encoding="utf-8")
    return values
