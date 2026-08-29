#!/usr/bin/env python3
"""Audit the three numerical items left for coauthor confirmation.

This script is intentionally cheap: it does not train a policy or harvest new
adjoints.  It reconstructs the Table-6 analytical benchmark from the source,
checks the deterministic IID/edge/stress state samplers, and reconciles the
nested Z/zeta budgets across YAML configs and completed run summaries.

Run from the repository root, for example::

    python3 scripts/coauthor_experiment_audit.py --strict

Outputs are written under ``paper/coauthor_checks`` by default.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml


TABLE6_JOBS = (
    "t6_merton_cap_d2",
    "t6_merton_cap_d10",
    "t6_merton_cap_d100",
    "t6_merton_cap_d100_exact",
)
STATE_SPLIT_JOBS = (
    "t8_merton_cap_iid",
    "t8_merton_cap_wealth_edge",
    "t8_merton_cap_stress",
    "t8_lko_cap_iid",
    "t8_lko_cap_wealth_edge",
    "t8_lko_cap_factor_edge",
    "t8_lko_cap_stress",
)
CONSTRAINED_FACTOR_JOBS = (
    "t12_lko_short_d10",
    "t12_lko_short_d50",
    "t12_lko_short_d100",
)
PREDICTABLE_RETURN_JOBS = (
    "t7_affine_canonical",
    "t7_affine_hedging",
)
PROJECTION_BUDGET_JOBS = (
    *TABLE6_JOBS,
    *PREDICTABLE_RETURN_JOBS,
    *STATE_SPLIT_JOBS,
    *CONSTRAINED_FACTOR_JOBS,
)


class AuditFailure(RuntimeError):
    """Raised only at the very end when --strict is requested."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Table 6, diagnostic-state splits, and M_out x M_in metadata."
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="repository root (default: current directory)",
    )
    parser.add_argument(
        "--run-root",
        default="runs_paper_full_shift",
        help="completed-run root to inspect (default: runs_paper_full_shift)",
    )
    parser.add_argument(
        "--output",
        default="paper/coauthor_checks",
        help="directory for the generated report",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="device used only for cheap deterministic state regeneration",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a nonzero exit code when a hard consistency check fails",
    )
    parser.add_argument(
        "--allow-missing-runs",
        action="store_true",
        help="treat absent completed-run files as warnings instead of failures",
    )
    return parser.parse_args()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return value


def maximum_absolute(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.detach().abs().max().cpu())


def allclose_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return maximum_absolute(left - right)


def tensor_hash(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        array = np.ascontiguousarray(tensor.detach().cpu().double().numpy())
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def manifest_tags(repo: Path) -> dict[str, dict[str, Any]]:
    manifest = load_yaml(repo / "paper" / "paper_suite.yaml")
    result: dict[str, dict[str, Any]] = {}
    for row in manifest.get("jobs", []):
        if isinstance(row, dict) and row.get("name"):
            result[str(row["name"])] = dict(row.get("tags", {}))
    return result


def locate_config(repo: Path, run_root: Path, job: str) -> tuple[Path, dict[str, Any]]:
    candidates = (
        run_root / job / "resolved_config.yaml",
        repo / "paper" / "generated_configs" / f"{job}.yaml",
    )
    for path in candidates:
        if path.exists():
            return path, load_yaml(path)
    raise FileNotFoundError(
        f"No resolved/generated config found for {job}; tried: "
        + ", ".join(str(path) for path in candidates)
    )


def expected_state_file(run_dir: Path, evaluation: dict[str, Any]) -> Path:
    explicit = evaluation.get("states_file")
    if explicit:
        return Path(str(explicit)).expanduser()
    mode = str(evaluation.get("state_mode", "iid")).lower().replace("-", "_")
    role = str(evaluation.get("state_role", mode))
    return run_dir / f"evaluation_states_{role}.pt"


def rmse_from_norm(frame: pd.DataFrame, column: str, dimension: int) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(float)
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(values * values) / float(dimension)))


def rmse_scalar(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(float)
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(values * values)))


def compare_numbers(left: Any, right: Any, *, atol: float = 1e-12, rtol: float = 1e-10) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    try:
        a, b = float(left), float(right)
    except (TypeError, ValueError):
        return left == right
    return math.isclose(a, b, abs_tol=atol, rel_tol=rtol)


def table6_row_from_run(run_dir: Path, tags: dict[str, Any]) -> dict[str, Any]:
    frame = pd.read_csv(run_dir / "pointwise.csv")
    recovery = load_json(run_dir / "recovery_summary.json")
    dimension = int(tags.get("d", 1))
    return {
        "n": dimension,
        "row_type": tags.get("row_type", "learned"),
        "DPO_u_RMSE": rmse_from_norm(frame, "reference_risky_l2_to_exact", dimension),
        "QP_u_RMSE": rmse_from_norm(frame, "full_risky_l2_to_exact", dimension),
        "DPO_C_RMSE": rmse_scalar(frame, "reference_consumption_abs_to_exact"),
        "QP_C_RMSE": rmse_scalar(frame, "full_consumption_abs_to_exact"),
        "DPO_portfolio_KKT": recovery.get("reference_kkt_pg_mean"),
        "QP_portfolio_KKT": recovery.get("full_kkt_pg_mean"),
        "DPO_consumption_KKT": recovery.get("reference_consumption_kkt_pg_mean"),
        "QP_consumption_KKT": recovery.get("full_consumption_kkt_pg_mean"),
        "full_zero_action_l2": recovery.get("full_zero_action_l2_mean"),
    }


def audit_table6(
    repo: Path,
    run_root: Path,
    output: Path,
    tags_by_job: dict[str, dict[str, Any]],
    device: torch.device,
    issues: list[dict[str, str]],
    allow_missing_runs: bool,
) -> dict[str, Any]:
    from mf_revision.experiments.states import sample_evaluation_states
    from mf_revision.models import build_model
    from mf_revision.recovery.consumption import consumption_pg_residual
    from mf_revision.recovery.qp import solve_qp

    config_path, config = locate_config(repo, run_root, "t6_merton_cap_d100_exact")
    model_config = dict(config["model"])
    model = build_model(model_config, device=device, dtype=torch.float64)
    evaluation = dict(config.get("evaluation", {}))

    # Prefer the exact diagnostic states actually used by the completed run.
    saved_path = expected_state_file(
        run_root / "t6_merton_cap_d100_exact", evaluation
    )
    state_source = "deterministic regeneration"
    if saved_path.exists():
        payload = torch.load(saved_path, map_location=device, weights_only=False)
        states = payload["states"].to(device=device, dtype=torch.float64)
        tau = payload["tau"].to(device=device, dtype=torch.float64)
        state_source = str(saved_path)
    else:
        count = min(int(evaluation.get("states", 128)), 64)
        generator = torch.Generator(device=device.type)
        generator.manual_seed(int(evaluation.get("state_seed", config.get("seed", 12345))))
        states, tau = sample_evaluation_states(
            model,
            count,
            evaluation,
            generator=generator,
            device=device,
            dtype=torch.float64,
        )

    policy = model.analytical_policy().to(device=device, dtype=torch.float64)
    control = policy(states, tau)
    fields = model.exact_adjoint_fields(states, tau)
    risky, consumption = model.split_control(control)
    if consumption is None:
        raise RuntimeError("Table-6 exact model unexpectedly has no consumption block")
    wealth = states[:, 0:1]
    lower, upper = model.consumption_bounds(states)
    exact_rate = model.exact_consumption_rate(tau)
    sigma = model.state_diffusion(states, control)
    expected_z = torch.einsum("bs,bsq->bq", fields["P_wealth_row"], sigma)

    g, q = model.local_qp_coefficients(
        states,
        fields["lambda"],
        fields["P_wealth_row"],
        fields["zeta_wealth"],
    )
    qp = solve_qp(
        g,
        q,
        constraint=model.constraint,
        cap=model.leverage_cap,
        tolerance=1e-12,
        curvature_floor=1e-12,
    )
    recovered_consumption = model.recover_consumption(states, fields["lambda"])
    if recovered_consumption is None:
        raise RuntimeError("Exact consumption recovery unexpectedly returned None")

    numerical_checks = {
        "policy_equals_exact_field_control_max_abs": allclose_error(
            control, fields["control"]
        ),
        "consumption_equals_X_times_exact_rate_max_abs": allclose_error(
            consumption, wealth * exact_rate
        ),
        "PXX_plus_gamma_lambda_over_X_max_abs": maximum_absolute(
            fields["PXX"] + model.gamma * fields["lambda"] / wealth
        ),
        "Z_minus_Psigma_max_abs": allclose_error(fields["Z_wealth"], expected_z),
        "zeta_zero_max_abs": maximum_absolute(fields["zeta_wealth"]),
        "QP_recovered_risky_minus_exact_risky_max_abs": allclose_error(
            qp.control, risky
        ),
        "scalar_recovered_consumption_minus_exact_max_abs": allclose_error(
            recovered_consumption, consumption
        ),
        "exact_consumption_projected_gradient_residual_max": maximum_absolute(
            consumption_pg_residual(model, states, fields["lambda"][:, 0:1], consumption)
        ),
        "risky_nonnegativity_violation": max(0.0, -float(risky.min().detach().cpu())),
        "risky_leverage_violation": max(
            0.0,
            float((risky.sum(dim=1) - model.leverage_cap).max().detach().cpu()),
        ),
        "consumption_lower_violation": max(
            0.0, float((lower - consumption).max().detach().cpu())
        ),
        "consumption_upper_violation": max(
            0.0, float((consumption - upper).max().detach().cpu())
        ),
        "QP_projected_gradient_residual_max": maximum_absolute(
            qp.projected_gradient_residual
        ),
    }
    tolerance_by_check = {
        "policy_equals_exact_field_control_max_abs": 1e-12,
        "consumption_equals_X_times_exact_rate_max_abs": 1e-12,
        "PXX_plus_gamma_lambda_over_X_max_abs": 1e-11,
        "Z_minus_Psigma_max_abs": 1e-11,
        "zeta_zero_max_abs": 1e-14,
        "QP_recovered_risky_minus_exact_risky_max_abs": 1e-8,
        "scalar_recovered_consumption_minus_exact_max_abs": 1e-10,
        "exact_consumption_projected_gradient_residual_max": 1e-8,
        "risky_nonnegativity_violation": 1e-12,
        "risky_leverage_violation": 1e-10,
        "consumption_lower_violation": 1e-12,
        "consumption_upper_violation": 1e-12,
        "QP_projected_gradient_residual_max": 1e-8,
    }
    check_status = {
        key: numerical_checks[key] <= tolerance_by_check[key]
        for key in numerical_checks
    }
    for key, passed in check_status.items():
        if not passed:
            issues.append(
                {
                    "severity": "FAIL",
                    "area": "Table 6 analytical construction",
                    "message": (
                        f"{key}={numerical_checks[key]:.6e} exceeds "
                        f"{tolerance_by_check[key]:.6e}"
                    ),
                }
            )

    recomputed_rows: list[dict[str, Any]] = []
    for job in TABLE6_JOBS:
        run_dir = run_root / job
        required = (run_dir / "pointwise.csv", run_dir / "recovery_summary.json")
        if not all(path.exists() for path in required):
            severity = "WARN" if allow_missing_runs else "FAIL"
            issues.append(
                {
                    "severity": severity,
                    "area": "Table 6 completed runs",
                    "message": f"Missing completed output for {job} under {run_root}",
                }
            )
            continue
        recomputed_rows.append(table6_row_from_run(run_dir, tags_by_job.get(job, {})))

    recomputed_frame = pd.DataFrame(recomputed_rows)
    if not recomputed_frame.empty:
        recomputed_frame = recomputed_frame.sort_values(["n", "row_type"])
        recomputed_frame.to_csv(output / "table6_recomputed.csv", index=False)

    collected_path = repo / "paper" / "collected_full_shift" / "table6_consumption_cap.csv"
    table_consistency: dict[str, Any] = {
        "collected_path": str(collected_path),
        "available": collected_path.exists(),
        "rows_checked": 0,
        "maximum_absolute_difference": 0.0,
        "consistent": None,
    }
    if collected_path.exists() and not recomputed_frame.empty and run_root.name == "runs_paper_full_shift":
        collected = pd.read_csv(collected_path)
        keys = ["n", "row_type"]
        merged = recomputed_frame.merge(
            collected,
            on=keys,
            how="outer",
            suffixes=("_recomputed", "_collected"),
            indicator=True,
        )
        columns = [column for column in recomputed_frame.columns if column not in keys]
        maximum = 0.0
        consistent = bool((merged["_merge"] == "both").all())
        for _, row in merged.iterrows():
            if row["_merge"] != "both":
                continue
            for column in columns:
                left, right = row[f"{column}_recomputed"], row[f"{column}_collected"]
                if pd.isna(left) and pd.isna(right):
                    continue
                if not compare_numbers(left, right):
                    consistent = False
                if not pd.isna(left) and not pd.isna(right):
                    maximum = max(maximum, abs(float(left) - float(right)))
        table_consistency.update(
            {
                "rows_checked": int((merged["_merge"] == "both").sum()),
                "maximum_absolute_difference": maximum,
                "consistent": consistent,
            }
        )
        if not consistent:
            issues.append(
                {
                    "severity": "FAIL",
                    "area": "Table 6 collector",
                    "message": "Collected Table 6 CSV does not match pointwise/recovery outputs",
                }
            )

    result = {
        "config_path": str(config_path),
        "state_source": state_source,
        "state_count": int(states.shape[0]),
        "device": str(device),
        "implementation": {
            "risky_reference": (
                "MertonModel._solve_exact_risky: exact constrained QP of alpha against "
                "gamma*Sigma over the configured simplex/orthant"
            ),
            "value_coefficient": (
                "MertonModel._build_value_coefficient_grid: solve_ivp for the CRRA "
                "value coefficient with clipped proportional consumption rate"
            ),
            "consumption_reference": (
                "C*(t,x)=x*A(tau)^(-1/gamma), clipped to "
                "[consumption_rate_min*x, consumption_rate_max*x]"
            ),
            "exact_adjoint_fields": (
                "lambda_X=A(tau)x^(-gamma), PXX=-gamma*A(tau)x^(-gamma-1), "
                "Z_X=PXX*sigma_X, zeta_X=0"
            ),
            "exact_policy_pipeline": (
                "policy.kind=analytic; the analytical policy supplies the rollout, while "
                "OL-BPTT and shifted-input estimation are still rerun numerically"
            ),
        },
        "numerical_checks": numerical_checks,
        "check_tolerances": tolerance_by_check,
        "check_status": check_status,
        "table6_collector_consistency": table_consistency,
        "recomputed_rows": recomputed_rows,
    }
    json_dump(output / "table6_construction_audit.json", result)
    return result


def state_mode_checks(
    states: torch.Tensor,
    tau: torch.Tensor,
    model: Any,
    evaluation: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    mode = str(evaluation.get("state_mode", "iid")).lower().replace("-", "_")
    count = int(states.shape[0])
    wealth_low, wealth_high = map(float, model.wealth_range)
    horizon = float(model.T)
    minimum_tau = float(evaluation.get("minimum_tau", max(horizon / 100.0, 1e-4)))
    edge_fraction = min(max(float(evaluation.get("edge_fraction", 0.05)), 1e-4), 0.49)
    failures: list[str] = []
    eps = 2e-12

    finite = bool(torch.isfinite(states).all() and torch.isfinite(tau).all())
    if not finite:
        failures.append("states/tau contain non-finite values")
    if float(states[:, 0].min().cpu()) < wealth_low - eps:
        failures.append("wealth falls below configured range")
    if float(states[:, 0].max().cpu()) > wealth_high + eps:
        failures.append("wealth exceeds configured range")
    if float(tau.min().cpu()) < -eps:
        failures.append("tau is negative")
    if float(tau.max().cpu()) > horizon + eps:
        failures.append("tau exceeds horizon")

    half = (count + 1) // 2
    if mode in {"wealth_edge", "wealth"}:
        lower_limit = wealth_low + edge_fraction * (wealth_high - wealth_low)
        upper_limit = wealth_high - edge_fraction * (wealth_high - wealth_low)
        if not bool((states[:half, 0] <= lower_limit + eps).all()):
            failures.append("lower half is not confined to the lower wealth-edge band")
        if not bool((states[half:, 0] >= upper_limit - eps).all()):
            failures.append("upper half is not confined to the upper wealth-edge band")
    elif mode in {"factor_edge", "factor"}:
        if model.dims.factor <= 0:
            failures.append("factor_edge was requested for a factor-free model")
        else:
            values = torch.as_tensor(
                evaluation.get("factor_values", [-1.5, 1.5]),
                device=states.device,
                dtype=states.dtype,
            ).flatten()
            expected = values[torch.arange(count, device=states.device) % values.numel()]
            expected = expected.view(-1, 1).expand(-1, model.dims.factor)
            if maximum_absolute(states[:, 1:] - expected) > eps:
                failures.append("factor-edge values do not match the configured alternating pattern")
    elif mode in {"stress", "joint_stress"}:
        expected_wealth = torch.where(
            (torch.arange(count, device=states.device) % 2).view(-1, 1) == 0,
            torch.full((count, 1), wealth_low, device=states.device, dtype=states.dtype),
            torch.full((count, 1), wealth_high, device=states.device, dtype=states.dtype),
        )
        expected_tau = torch.where(
            (torch.arange(count, device=states.device) % 4 < 2).view(-1, 1),
            torch.full((count, 1), minimum_tau, device=states.device, dtype=states.dtype),
            torch.full((count, 1), horizon, device=states.device, dtype=states.dtype),
        )
        if maximum_absolute(states[:, 0:1] - expected_wealth) > eps:
            failures.append("stress wealth values do not alternate between range endpoints")
        if maximum_absolute(tau - expected_tau) > eps:
            failures.append("stress tau values do not follow the minimum/horizon pattern")
        if model.dims.factor:
            values = torch.as_tensor(
                evaluation.get("factor_values", [-1.5, 1.5]),
                device=states.device,
                dtype=states.dtype,
            ).flatten()
            expected = values[torch.arange(count, device=states.device) % values.numel()]
            expected = expected.view(-1, 1).expand(-1, model.dims.factor)
            if maximum_absolute(states[:, 1:] - expected) > eps:
                failures.append("stress factors do not match the configured alternating pattern")

    factor = states[:, 1:] if model.dims.factor else None
    statistics: dict[str, Any] = {
        "mode": mode,
        "count": count,
        "wealth_min": float(states[:, 0].min().cpu()),
        "wealth_max": float(states[:, 0].max().cpu()),
        "tau_min": float(tau.min().cpu()),
        "tau_max": float(tau.max().cpu()),
        "wealth_range_config": [wealth_low, wealth_high],
        "horizon": horizon,
        "edge_fraction": edge_fraction,
        "minimum_tau_for_edge_or_stress": minimum_tau,
        "factor_dim": int(model.dims.factor),
        "factor_min": None if factor is None else float(factor.min().cpu()),
        "factor_max": None if factor is None else float(factor.max().cpu()),
        "state_hash_sha256": tensor_hash(states, tau),
        "passed": not failures,
    }
    return statistics, failures


def audit_state_splits(
    repo: Path,
    run_root: Path,
    output: Path,
    device: torch.device,
    issues: list[dict[str, str]],
) -> list[dict[str, Any]]:
    from mf_revision.experiments.states import sample_evaluation_states
    from mf_revision.models import build_model

    rows: list[dict[str, Any]] = []
    for job in STATE_SPLIT_JOBS:
        config_path, config = locate_config(repo, run_root, job)
        evaluation = dict(config.get("evaluation", {}))
        model = build_model(dict(config["model"]), device=device, dtype=torch.float64)
        count = int(evaluation.get("states", 128))
        generator = torch.Generator(device=device.type)
        generator.manual_seed(int(evaluation.get("state_seed", config.get("seed", 12345))))
        generated_states, generated_tau = sample_evaluation_states(
            model,
            count,
            evaluation,
            generator=generator,
            device=device,
            dtype=torch.float64,
        )

        saved_path = expected_state_file(run_root / job, evaluation)
        saved_available = saved_path.exists()
        regeneration_match: bool | None = None
        if saved_available:
            payload = torch.load(saved_path, map_location=device, weights_only=False)
            states = payload["states"].to(device=device, dtype=torch.float64)
            tau = payload["tau"].to(device=device, dtype=torch.float64)
            if states.shape == generated_states.shape and tau.shape == generated_tau.shape:
                regeneration_match = bool(
                    torch.equal(states, generated_states) and torch.equal(tau, generated_tau)
                )
            else:
                regeneration_match = False
        else:
            states, tau = generated_states, generated_tau

        statistics, failures = state_mode_checks(states, tau, model, evaluation)
        if failures:
            for failure in failures:
                issues.append(
                    {
                        "severity": "FAIL",
                        "area": "Diagnostic-state sampling",
                        "message": f"{job}: {failure}",
                    }
                )
        if saved_available and regeneration_match is False:
            # CPU and CUDA RNG streams are not expected to match.  The saved-state
            # properties above are the authoritative check, so this is a warning.
            issues.append(
                {
                    "severity": "WARN",
                    "area": "Diagnostic-state reproducibility",
                    "message": (
                        f"{job}: saved states differ from regeneration on {device}; "
                        "this can occur when the original run used a different RNG device"
                    ),
                }
            )

        row = {
            "job": job,
            "config_path": str(config_path),
            "state_mode": statistics["mode"],
            "state_role": evaluation.get("state_role", statistics["mode"]),
            "count": statistics["count"],
            "wealth_min": statistics["wealth_min"],
            "wealth_max": statistics["wealth_max"],
            "tau_min": statistics["tau_min"],
            "tau_max": statistics["tau_max"],
            "factor_dim": statistics["factor_dim"],
            "factor_min": statistics["factor_min"],
            "factor_max": statistics["factor_max"],
            "edge_fraction": statistics["edge_fraction"],
            "minimum_tau": statistics["minimum_tau_for_edge_or_stress"],
            "saved_state_file": str(saved_path),
            "saved_state_available": saved_available,
            "regeneration_matches_saved_exactly": regeneration_match,
            "state_hash_sha256": statistics["state_hash_sha256"],
            "passed": statistics["passed"],
            "failures": "; ".join(failures),
        }
        rows.append(row)

    pd.DataFrame(rows).to_csv(output / "state_sampling_audit.csv", index=False)
    json_dump(output / "state_sampling_audit.json", rows)
    return rows


def audit_projection_budgets(
    repo: Path,
    run_root: Path,
    output: Path,
    tags_by_job: dict[str, dict[str, Any]],
    issues: list[dict[str, str]],
    allow_missing_runs: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in PROJECTION_BUDGET_JOBS:
        config_path, config = locate_config(repo, run_root, job)
        evaluation = dict(config.get("evaluation", {}))
        outer = int(evaluation.get("z_outer_paths", evaluation.get("continuations", 0)))
        inner = int(evaluation.get("z_inner_paths", 1))
        continuations = int(evaluation.get("continuations", 0))
        antithetic = bool(evaluation.get("antithetic", False))
        expected_total = outer * inner
        expected_units = outer // 2 if antithetic else outer

        summary_path = run_root / job / "adjoint_summary.json"
        summary = load_json(summary_path) if summary_path.exists() else {}
        if not summary_path.exists():
            severity = "WARN" if allow_missing_runs else "FAIL"
            issues.append(
                {
                    "severity": severity,
                    "area": "Projection budget metadata",
                    "message": f"Missing adjoint_summary.json for {job} under {run_root}",
                }
            )

        inner_match = (
            None
            if not summary
            else int(summary.get("projection_inner_paths", -1)) == inner
        )
        total_match = (
            None
            if not summary
            else int(summary.get("projection_total_future_rollouts", -1)) == expected_total
        )
        unit_value = summary.get("projection_independent_mc_units") if summary else None
        units_match = (
            None if unit_value is None else int(unit_value) == expected_units
        )
        continuation_match = (
            None
            if not summary
            else int(summary.get("continuations", -1)) == continuations
        )
        for label, matched in (
            ("M_in", inner_match),
            ("M_out*M_in", total_match),
            ("independent outer units", units_match),
            ("level continuations", continuation_match),
        ):
            if matched is False:
                issues.append(
                    {
                        "severity": "FAIL",
                        "area": "Projection budget metadata",
                        "message": f"{job}: summary does not match config for {label}",
                    }
                )

        row = {
            "job": job,
            "table": tags_by_job.get(job, {}).get("table"),
            "dimension": tags_by_job.get(job, {}).get("d", config.get("model", {}).get("d")),
            "state_mode": evaluation.get("state_mode", "iid"),
            "config_path": str(config_path),
            "level_continuations_M": continuations,
            "M_out": outer,
            "M_in": inner,
            "M_out_times_M_in": expected_total,
            "antithetic": antithetic,
            "independent_outer_units_expected": expected_units,
            "summary_available": bool(summary),
            "summary_projection_inner_paths": summary.get("projection_inner_paths"),
            "summary_projection_total_future_rollouts": summary.get(
                "projection_total_future_rollouts"
            ),
            "summary_projection_independent_mc_units": summary.get(
                "projection_independent_mc_units"
            ),
            "summary_continuations": summary.get("continuations"),
            "inner_match": inner_match,
            "total_match": total_match,
            "independent_units_match": units_match,
            "continuations_match": continuation_match,
        }
        rows.append(row)

    pd.DataFrame(rows).to_csv(output / "projection_budget_audit.csv", index=False)
    json_dump(output / "projection_budget_audit.json", rows)
    return rows


def markdown_report(
    repo: Path,
    run_root: Path,
    table6: dict[str, Any],
    state_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    issues: list[dict[str, str]],
) -> str:
    failures = [item for item in issues if item["severity"] == "FAIL"]
    warnings = [item for item in issues if item["severity"] == "WARN"]
    overall = "PASS" if not failures else "FAIL"
    lines: list[str] = [
        "# Coauthor experiment audit",
        "",
        f"- Overall: **{overall}**",
        f"- Repository: `{repo}`",
        f"- Inspected run root: `{run_root}`",
        f"- Hard failures: {len(failures)}",
        f"- Warnings: {len(warnings)}",
        "",
        "## 1. Table 6 analytical/reference construction",
        "",
        "The source and numerical identity checks support the following construction:",
        "",
        "- risky reference: exact constrained Merton QP over the configured simplex;",
        "- value coefficient: CRRA coefficient ODE solved by `solve_ivp`;",
        "- consumption reference: proportional rate `A(tau)^(-1/gamma)` clipped to the configured cap;",
        "- exact fields: `lambda_X=A(tau)X^(-gamma)`, `PXX=-gamma*lambda_X/X`, `Z_X=PXX*sigma_X`, and benchmark `zeta_X=0`;",
        "- the exact-policy row still runs the numerical OL-BPTT/shift/recovery pipeline; only the reference actor is analytical.",
        "",
        f"State source: `{table6['state_source']}` ({table6['state_count']} states).",
        "",
        "| Check | Error | Pass |",
        "|---|---:|:---:|",
    ]
    for name, value in table6["numerical_checks"].items():
        lines.append(f"| `{name}` | {value:.6e} | {'yes' if table6['check_status'][name] else 'NO'} |")
    consistency = table6["table6_collector_consistency"]
    lines.extend(
        [
            "",
            "Collector consistency:",
            "",
            f"- available: `{consistency['available']}`",
            f"- rows checked: `{consistency['rows_checked']}`",
            f"- maximum absolute difference: `{consistency['maximum_absolute_difference']:.6e}`",
            f"- consistent: `{consistency['consistent']}`",
            "",
            "## 2. IID / edge / stress state definitions",
            "",
            "| Job | Mode | N | Wealth range observed | Tau range observed | Factor range observed | Saved states | Pass |",
            "|---|---|---:|---|---|---|:---:|:---:|",
        ]
    )
    for row in state_rows:
        factor_range = (
            "–"
            if row["factor_dim"] == 0
            else f"[{row['factor_min']:.6g}, {row['factor_max']:.6g}]"
        )
        lines.append(
            f"| `{row['job']}` | `{row['state_mode']}` | {row['count']} | "
            f"[{row['wealth_min']:.6g}, {row['wealth_max']:.6g}] | "
            f"[{row['tau_min']:.6g}, {row['tau_max']:.6g}] | {factor_range} | "
            f"{'yes' if row['saved_state_available'] else 'no'} | "
            f"{'yes' if row['passed'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "The code-level definitions are:",
            "",
            "- `iid`: the model's canonical initial-state sampler;",
            "- `wealth_edge`: half in the bottom 5% wealth band and half in the top 5% band;",
            "- `factor_edge`: alternating factor values `-1.5, +1.5`;",
            "- `stress`: wealth at the two endpoints, tau at `max(T/100,1e-4)` or `T`, and factors alternating `-1.5, +1.5` when present.",
            "",
            "## 3. Nested shifted-input budgets",
            "",
            "| Job | Table | d | M | M_out x M_in | Independent outer units | Summary match |",
            "|---|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in budget_rows:
        matches = [
            row["inner_match"],
            row["total_match"],
            row["independent_units_match"],
            row["continuations_match"],
        ]
        match_text = "n/a" if all(value is None for value in matches) else (
            "yes" if all(value is not False for value in matches) else "NO"
        )
        lines.append(
            f"| `{row['job']}` | {row['table'] or '–'} | {row['dimension'] or '–'} | "
            f"{row['level_continuations_M']} | {row['M_out']} x {row['M_in']} | "
            f"{row['independent_outer_units_expected']} | {match_text} |"
        )

    lines.extend(["", "## 4. Issues", ""])
    if not issues:
        lines.append("No issues were found.")
    else:
        for item in issues:
            lines.append(
                f"- **{item['severity']}** — {item['area']}: {item['message']}"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A PASS means the three coauthor-confirmation items can be documented from the current source, resolved configs, and run summaries without launching a new paper-scale experiment. A separate full-budget rerun is useful only as an independent reproducibility check.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    if not (repo / "src" / "mf_revision").exists():
        raise SystemExit(f"Not a compatible repository: {repo}")
    run_root = Path(args.run_root).expanduser()
    if not run_root.is_absolute():
        run_root = (repo / run_root).resolve()
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = (repo / output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo / "src"))
    device = choose_device(args.device)
    tags_by_job = manifest_tags(repo)
    issues: list[dict[str, str]] = []

    table6 = audit_table6(
        repo,
        run_root,
        output,
        tags_by_job,
        device,
        issues,
        args.allow_missing_runs,
    )
    state_rows = audit_state_splits(repo, run_root, output, device, issues)
    budget_rows = audit_projection_budgets(
        repo,
        run_root,
        output,
        tags_by_job,
        issues,
        args.allow_missing_runs,
    )

    report = markdown_report(repo, run_root, table6, state_rows, budget_rows, issues)
    (output / "COAUTHOR_EXPERIMENT_AUDIT.md").write_text(report, encoding="utf-8")
    json_dump(
        output / "audit_manifest.json",
        {
            "repo": str(repo),
            "run_root": str(run_root),
            "device": str(device),
            "overall": "FAIL" if any(item["severity"] == "FAIL" for item in issues) else "PASS",
            "issues": issues,
            "files": [
                "COAUTHOR_EXPERIMENT_AUDIT.md",
                "table6_construction_audit.json",
                "table6_recomputed.csv",
                "state_sampling_audit.csv",
                "state_sampling_audit.json",
                "projection_budget_audit.csv",
                "projection_budget_audit.json",
            ],
        },
    )

    print(report)
    print(f"\n[written] {output / 'COAUTHOR_EXPERIMENT_AUDIT.md'}")
    hard_failures = [item for item in issues if item["severity"] == "FAIL"]
    if args.strict and hard_failures:
        raise AuditFailure(f"{len(hard_failures)} hard audit failure(s)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditFailure as error:
        print(f"[audit failed] {error}", file=sys.stderr)
        raise SystemExit(2)
