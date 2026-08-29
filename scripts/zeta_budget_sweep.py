#!/usr/bin/env python3
"""MC-budget sweep for the exact-policy predictable-return shift estimator.

This is a post-training audit. It keeps the saved level adjoints (lambda, P),
evaluation states, and analytic policy fixed, and re-estimates only the Brownian
projection / shifted wealth-row input zeta^X at several (outer, inner) budgets.

For each budget and replicate it reports:
  * magnitude of the estimated wealth-row shift;
  * a fixed-anchor regression SNR (P sigma is treated as fixed here);
  * full-shift decoded-policy RMSE vs the analytical policy;
  * zero-shift oracle policy RMSE vs the analytical policy;
  * the raw P^{XY} vs analytical V_{XY} nRMSE (constant across budgets).

Default runs:
  runs_paper_full_shift/t7_affine_canonical
  runs_paper_full_shift/t7_affine_hedging

Example
-------
cd /path/to/pgdpo-adjoint-control-recovery
PYTHONPATH=src python scripts/zeta_budget_sweep.py \
  --budgets 256x8,512x16,1024x32 \
  --replicates 3

Notes
-----
* No policy training is performed.
* The expensive level-adjoint harvest is NOT repeated. The saved P is used as
  the control-variate anchor, so this audit isolates the finite-MC behavior of
  the shifted-input estimator.
* The SNR printed here is conditional on the saved P anchor (p_sigma_se=0), so
  it should not be numerically equated with the manuscript's full-pipeline SNR.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
import yaml

from mf_revision.adjoints.harvest import _projection_units_one_state
from mf_revision.adjoints.projection import estimate_brownian_projection
from mf_revision.config import ExperimentConfig
from mf_revision.models import build_model
from mf_revision.policies import build_policy
from mf_revision.random import BrownianSpec, DeterministicBrownianBank
from mf_revision.recovery import recover_controls
from mf_revision.runtime import resolve_device, resolve_dtype, set_global_seed
from mf_revision.types import AdjointEstimate


RUNS = {
    "canonical": "t7_affine_canonical",
    "hedging": "t7_affine_hedging",
}


def parse_budget(text: str) -> tuple[int, int]:
    value = text.strip().lower().replace("*", "x")
    if "x" not in value:
        raise argparse.ArgumentTypeError(
            f"Budget must have OUTERxINNER form, e.g. 512x16; got {text!r}"
        )
    outer_s, inner_s = value.split("x", 1)
    outer, inner = int(outer_s), int(inner_s)
    if outer <= 0 or outer % 2:
        raise argparse.ArgumentTypeError("OUTER must be positive and even")
    if inner <= 0 or (inner > 1 and inner % 2):
        raise argparse.ArgumentTypeError(
            "INNER must be 1 or a positive even integer"
        )
    return outer, inner


def rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x.double().square())).detach().cpu())


def vector_rms(x: torch.Tensor) -> float:
    return float(
        torch.sqrt(torch.mean(torch.sum(x.double().square(), dim=-1))).detach().cpu()
    )


def coordinate_rmse(estimate: torch.Tensor, target: torch.Tensor) -> float:
    return rms(estimate - target)


def nrmse(estimate: torch.Tensor, target: torch.Tensor) -> float:
    denominator = torch.sqrt(torch.mean(target.double().square())).clamp_min(1e-30)
    numerator = torch.sqrt(torch.mean((estimate.double() - target.double()).square()))
    return float((numerator / denominator).detach().cpu())


def load_resolved_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    # resolved_config.yaml contains the ordinary ExperimentConfig payload.
    return ExperimentConfig.from_mapping(raw, source_path=str(path))


def build_projection_estimate(
    *,
    model,
    policy,
    saved: AdjointEstimate,
    outer_paths: int,
    inner_paths: int,
    projection_seed: int,
    inner_seed: int,
    pair_batch: int,
    ridge: float,
    graph_mode: str,
    state_limit: int | None,
) -> tuple[AdjointEstimate, dict[str, float]]:
    count = saved.states.shape[0] if state_limit is None else min(state_limit, saved.states.shape[0])
    states = saved.states[:count]
    tau = saved.tau[:count]
    lambda_ = saved.lambda_[:count]
    p = saved.p[:count]
    sigma_ref = saved.sigma_ref[:count]
    reference_control = saved.reference_control[:count]

    state_ids_meta = saved.metadata.get("state_ids") if isinstance(saved.metadata, dict) else None
    if state_ids_meta is None:
        state_ids = list(range(count))
    else:
        state_ids = [int(v) for v in state_ids_meta[:count]]

    projection_bank = DeterministicBrownianBank(
        BrownianSpec(
            seed=int(projection_seed),
            continuations=int(outer_paths),
            steps=model.n_steps,
            brownian_dim=model.dims.brownian,
            antithetic=True,
            dtype=states.dtype,
            pairing="first_step_common_future",
        )
    )

    z_rows: list[torch.Tensor] = []
    zeta_rows: list[torch.Tensor] = []
    z_se_rows: list[torch.Tensor] = []
    zeta_se_rows: list[torch.Tensor] = []
    gram_conditions: list[float] = []

    started = time.perf_counter()
    for i in range(count):
        p_sigma = p[i] @ sigma_ref[i]
        # This audit conditions on the already-harvested P anchor.  Hence the
        # anchor uncertainty is deliberately set to zero here; replicate
        # variation below measures projection MC error directly.
        p_sigma_se = torch.zeros_like(p_sigma)

        response, brownian, dt_value = _projection_units_one_state(
            model,
            policy,
            states[i : i + 1],
            tau[i : i + 1],
            state_id=state_ids[i],
            bank=projection_bank,
            continuation_batch=2 * int(pair_batch),
            inner_paths=int(inner_paths),
            inner_seed=int(inner_seed),
            graph_mode=graph_mode,
        )
        result = estimate_brownian_projection(
            response,
            brownian,
            dt=dt_value,
            p_sigma=p_sigma,
            p_sigma_se=p_sigma_se,
            method="ols_control_variate",
            ridge=float(ridge),
        )
        z_rows.append(result.z)
        zeta_rows.append(result.zeta)
        z_se_rows.append(result.z_se)
        zeta_se_rows.append(result.zeta_se)
        gram_conditions.append(float(result.diagnostics["projection_gram_condition"]))

    z = torch.stack(z_rows)
    zeta = torch.stack(zeta_rows)
    z_se = torch.stack(z_se_rows)
    zeta_se = torch.stack(zeta_se_rows)

    metadata = copy.deepcopy(saved.metadata)
    metadata.update(
        {
            "audit": "fixed_P_shift_budget_sweep",
            "projection_continuations": int(outer_paths),
            "projection_inner_paths": int(inner_paths),
            "projection_seed": int(projection_seed),
            "projection_inner_seed": int(inner_seed),
            "projection_pair_batch": int(pair_batch),
            "projection_ridge": float(ridge),
            "anchor_uncertainty_included": False,
        }
    )
    estimate = replace(
        saved,
        states=states,
        tau=tau,
        lambda_=lambda_,
        p=p,
        z=z,
        zeta=zeta,
        sigma_ref=sigma_ref,
        reference_control=reference_control,
        lambda_se=(saved.lambda_se[:count] if saved.lambda_se is not None else None),
        p_se=(saved.p_se[:count] if saved.p_se is not None else None),
        z_se=z_se,
        zeta_se=zeta_se,
        metadata=metadata,
    )
    return estimate, {
        "projection_seconds": time.perf_counter() - started,
        "gram_condition_mean": float(sum(gram_conditions) / len(gram_conditions)),
        "gram_condition_max": float(max(gram_conditions)),
    }


def one_run(
    root: Path,
    run_name: str,
    *,
    outer_paths: int,
    inner_paths: int,
    replicate: int,
    state_limit: int | None,
    seed_stride: int,
    device_override: str | None,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    run_dir = root / "runs_paper_full_shift" / run_name
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    config = load_resolved_config(run_dir / "resolved_config.yaml")
    if device_override is not None:
        config.device = device_override
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype)
    set_global_seed(config.seed)

    model = build_model(config.model, device=device, dtype=dtype)
    policy = build_policy(model, config.policy, device=device, dtype=dtype)
    policy.eval()

    saved = AdjointEstimate.load(run_dir / "adjoints.pt", map_location=device)
    # Coerce loaded tensors to the configured dtype in case the runtime default changed.
    tensor_fields = [
        "states", "tau", "lambda_", "p", "z", "zeta", "sigma_ref",
        "reference_control", "lambda_se", "p_se", "z_se", "zeta_se",
    ]
    kwargs = {}
    for name in tensor_fields:
        value = getattr(saved, name)
        kwargs[name] = None if value is None else value.to(device=device, dtype=dtype)
    saved = replace(saved, **kwargs)

    evaluation = config.evaluation
    base_projection_seed = int(
        evaluation.get("z_brownian_seed", evaluation.get("brownian_seed", config.seed + 2000) + 104729)
    )
    base_inner_seed = int(evaluation.get("z_inner_seed", base_projection_seed + 130363))
    if replicate == 0:
        projection_seed = base_projection_seed
        inner_seed = base_inner_seed
    else:
        projection_seed = base_projection_seed + int(seed_stride) * replicate
        inner_seed = base_inner_seed + int(seed_stride) * replicate

    estimate, projection_info = build_projection_estimate(
        model=model,
        policy=policy,
        saved=saved,
        outer_paths=outer_paths,
        inner_paths=inner_paths,
        projection_seed=projection_seed,
        inner_seed=inner_seed,
        pair_batch=int(evaluation.get("z_outer_pair_batch", 32)),
        ridge=float(evaluation.get("z_ridge_relative", 1e-12)),
        graph_mode=str(evaluation.get("graph_mode", "ol")).lower(),
        state_limit=state_limit,
    )

    recovery = recover_controls(model, estimate, config.recovery)
    exact = model.exact_adjoint_fields(estimate.states, estimate.tau)
    target_control = exact["control"]
    vxy = exact["VXY"]
    pxy = estimate.p[:, 0, 1:2]
    exact_z = exact["Z_wealth"]

    zeta_x = estimate.zeta_x
    zeta_se_x = estimate.zeta_se[:, 0, :]
    statewise_snr = torch.linalg.norm(zeta_x, dim=-1) / torch.linalg.norm(
        zeta_se_x, dim=-1
    ).clamp_min(1e-14)

    baseline_zeta_diff = float("nan")
    baseline_outer = int(evaluation.get("z_outer_paths", evaluation.get("continuations", 8192)))
    baseline_inner = int(evaluation.get("z_inner_paths", 8))
    if replicate == 0 and outer_paths == baseline_outer and inner_paths == baseline_inner:
        baseline = saved.zeta[: estimate.states.shape[0], 0, :]
        baseline_zeta_diff = float((zeta_x - baseline).abs().max().detach().cpu())

    summary: dict[str, float | int | str] = {
        "run": run_name,
        "budget": f"{outer_paths}x{inner_paths}",
        "outer_paths": int(outer_paths),
        "outer_pairs": int(outer_paths // 2),
        "inner_paths": int(inner_paths),
        "total_future_rollouts_per_state": int(outer_paths * inner_paths),
        "replicate": int(replicate),
        "states": int(estimate.states.shape[0]),
        "projection_seed": int(projection_seed),
        "inner_seed": int(inner_seed),
        "PXY_vs_VXY_nRMSE": nrmse(pxy, vxy),
        "zetaX_vector_RMS": vector_rms(zeta_x),
        "zetaX_to_exactZ_RMS_ratio": vector_rms(zeta_x) / max(vector_rms(exact_z), 1e-30),
        "fixed_anchor_mean_statewise_SNR": float(statewise_snr.mean().detach().cpu()),
        "full_shift_policy_RMSE": coordinate_rmse(recovery.full_control, target_control),
        "zero_shift_policy_RMSE": coordinate_rmse(recovery.zero_shift_control, target_control),
        "full_vs_zero_policy_RMSE": coordinate_rmse(
            recovery.full_control, recovery.zero_shift_control
        ),
        "baseline_zeta_max_abs_diff": baseline_zeta_diff,
        **projection_info,
    }

    pointwise = pd.DataFrame(
        {
            "run": run_name,
            "budget": f"{outer_paths}x{inner_paths}",
            "replicate": replicate,
            "state_index": range(estimate.states.shape[0]),
            "wealth": estimate.states[:, 0].detach().cpu().numpy(),
            "factor": estimate.states[:, 1].detach().cpu().numpy(),
            "tau": estimate.tau[:, 0].detach().cpu().numpy(),
            "PXY": pxy[:, 0].detach().cpu().numpy(),
            "VXY": vxy[:, 0].detach().cpu().numpy(),
            "zeta_norm": torch.linalg.norm(zeta_x, dim=-1).detach().cpu().numpy(),
            "zeta_se_norm_fixed_anchor": torch.linalg.norm(zeta_se_x, dim=-1).detach().cpu().numpy(),
            "zeta_SNR_fixed_anchor": statewise_snr.detach().cpu().numpy(),
            "full_policy": recovery.full_control[:, 0].detach().cpu().numpy(),
            "zero_policy": recovery.zero_shift_control[:, 0].detach().cpu().numpy(),
            "exact_policy": target_control[:, 0].detach().cpu().numpy(),
        }
    )
    return summary, pointwise


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "PXY_vs_VXY_nRMSE",
        "zetaX_vector_RMS",
        "zetaX_to_exactZ_RMS_ratio",
        "fixed_anchor_mean_statewise_SNR",
        "full_shift_policy_RMSE",
        "zero_shift_policy_RMSE",
        "full_vs_zero_policy_RMSE",
        "projection_seconds",
        "gram_condition_mean",
        "gram_condition_max",
    ]
    rows = []
    for (run, budget), frame in raw.groupby(["run", "budget"], sort=False):
        row = {
            "run": run,
            "budget": budget,
            "outer_paths": int(frame["outer_paths"].iloc[0]),
            "inner_paths": int(frame["inner_paths"].iloc[0]),
            "replicates": int(len(frame)),
            "states": int(frame["states"].iloc[0]),
        }
        for metric in metrics:
            values = frame[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--runs",
        default="canonical,hedging",
        help="Comma-separated aliases canonical,hedging or exact t7 run names",
    )
    parser.add_argument(
        "--budgets",
        default="256x8,512x16,1024x32",
        help="Comma-separated OUTERxINNER budgets",
    )
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument(
        "--states",
        type=int,
        default=None,
        help="Optional state count for a smoke test; default uses all saved states",
    )
    parser.add_argument("--seed-stride", type=int, default=1000003)
    parser.add_argument("--device", default=None, help="Override config device, e.g. cpu or cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/zeta_budget_sweep"),
    )
    args = parser.parse_args()

    if args.replicates <= 0:
        raise SystemExit("--replicates must be positive")
    if args.states is not None and args.states <= 0:
        raise SystemExit("--states must be positive")

    root = args.root.expanduser().resolve()
    output = args.output
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)

    run_tokens = [item.strip() for item in args.runs.split(",") if item.strip()]
    run_names = [RUNS.get(token, token) for token in run_tokens]
    budgets = [parse_budget(item) for item in args.budgets.split(",") if item.strip()]

    raw_rows: list[dict[str, float | int | str]] = []
    pointwise_frames: list[pd.DataFrame] = []

    print("Exact-policy predictable-return shift MC-budget sweep")
    print(f"root={root}")
    print(f"runs={run_names}")
    print(f"budgets={[f'{o}x{i}' for o, i in budgets]}, replicates={args.replicates}")
    if args.states is not None:
        print(f"SMOKE/partial mode: first {args.states} saved states")
    print()

    for run_name in run_names:
        for outer, inner in budgets:
            for rep in range(args.replicates):
                print(f"[{run_name}] budget={outer}x{inner} rep={rep}", flush=True)
                summary, pointwise = one_run(
                    root,
                    run_name,
                    outer_paths=outer,
                    inner_paths=inner,
                    replicate=rep,
                    state_limit=args.states,
                    seed_stride=args.seed_stride,
                    device_override=args.device,
                )
                raw_rows.append(summary)
                pointwise_frames.append(pointwise)
                print(
                    "  zeta/Z={:.4g}, full RMSE={:.4g}, zero RMSE={:.4g}, {:.1f}s".format(
                        summary["zetaX_to_exactZ_RMS_ratio"],
                        summary["full_shift_policy_RMSE"],
                        summary["zero_shift_policy_RMSE"],
                        summary["projection_seconds"],
                    ),
                    flush=True,
                )

    raw = pd.DataFrame(raw_rows)
    summary = aggregate(raw)
    pointwise = pd.concat(pointwise_frames, ignore_index=True)

    raw_path = output / "replicates.csv"
    summary_path = output / "summary.csv"
    pointwise_path = output / "pointwise.csv"
    json_path = output / "summary.json"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    pointwise.to_csv(pointwise_path, index=False)
    json_path.write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8"
    )

    print("\nAggregated summary")
    display_cols = [
        "run",
        "budget",
        "zetaX_to_exactZ_RMS_ratio_mean",
        "fixed_anchor_mean_statewise_SNR_mean",
        "full_shift_policy_RMSE_mean",
        "full_shift_policy_RMSE_sd",
        "zero_shift_policy_RMSE_mean",
    ]
    print(summary[display_cols].to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {raw_path}")
    print(f"Saved: {pointwise_path}")


if __name__ == "__main__":
    main()
