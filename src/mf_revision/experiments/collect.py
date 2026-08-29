"""Collect manifest runs into Section 5 and Appendix-ready CSV tables."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math
import pandas as pd

from .suite import PaperSuite


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _pointwise(path: Path) -> pd.DataFrame:
    source = path / "pointwise.csv"
    return pd.read_csv(source) if source.exists() else pd.DataFrame()


def _metric(row: pd.Series | dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = row[key]
    except (KeyError, TypeError):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rmse_from_norm(frame: pd.DataFrame, column: str, dimension: int = 1) -> float:
    if column not in frame or frame.empty:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(math.sqrt(float((values * values).mean()) / float(max(dimension, 1))))


def _rmse_scalar(frame: pd.DataFrame, column: str) -> float:
    return _rmse_from_norm(frame, column, 1)


def collect_paper_results(
    manifest_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    suite = PaperSuite(manifest_path)
    target = Path(output_dir or (suite.path.parent / "collected")).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for job in suite.jobs:
        run = suite.output_root / job.name
        tags = dict(job.tags or {})
        adjoint = _json(run / "adjoint_summary.json")
        recovery = _json(run / "recovery_summary.json")
        initialization = _json(run / "initialization_audit.json")
        graph = _json(run / "graph_ablation.json")
        timing: dict[str, Any] = {}
        for timing_file in run.glob("timing_*.json"):
            timing.update(_json(timing_file))
        record: dict[str, Any] = {
            "job": job.name,
            "stage": job.stage,
            "groups": ",".join(job.groups),
            "complete": bool(adjoint or recovery or initialization or graph),
            **tags,
        }
        record.update({f"adj_{key}": value for key, value in adjoint.items()})
        record.update({f"rec_{key}": value for key, value in recovery.items()})
        record.update({f"init_{key}": value for key, value in initialization.items()})
        record.update({f"graph_{key}": value for key, value in graph.items()})
        record.update({f"time_{key}": value for key, value in timing.items()})
        records.append(record)
        by_name[job.name] = {
            "job": job,
            "run": run,
            "tags": tags,
            "adjoint": adjoint,
            "recovery": recovery,
            "initialization": initialization,
            "graph": graph,
            "timing": timing,
            "pointwise": _pointwise(run),
        }
    all_frame = pd.DataFrame(records)
    all_path = target / "all_runs.csv"
    all_frame.to_csv(all_path, index=False)

    written: dict[str, str] = {"all_runs": str(all_path)}

    # Table 3: no-short-sale Merton, DPO / barrier / full-shift QP.
    table3_rows: list[dict[str, Any]] = []
    for value in by_name.values():
        if str(value["tags"].get("table")) != "3":
            continue
        frame, rec = value["pointwise"], value["recovery"]
        d = int(value["tags"].get("d", 1))
        for method, prefix in (("DPO", "reference"), ("B-PGDPO", "barrier"), ("QP-PGDPO", "full")):
            table3_rows.append(
                {
                    "n": d,
                    "method": method,
                    "coordinatewise_u_rmse": _rmse_from_norm(
                        frame, f"{prefix}_risky_l2_to_exact", d
                    ),
                    "mean_kkt_pg": _metric(rec, f"{prefix}_kkt_pg_mean"),
                    "H_gain_vs_DPO": 0.0 if method == "DPO" else _metric(
                        rec, f"{prefix}_gain_vs_reference_mean"
                    ),
                    "H_gap_to_QP": (
                        _metric(rec, "full_gain_vs_reference_mean")
                        if method == "DPO"
                        else (_metric(rec, "barrier_gap_to_full_qp_mean") if method == "B-PGDPO" else 0.0)
                    ),
                }
            )
    if table3_rows:
        path = target / "table3_noshort.csv"
        pd.DataFrame(table3_rows).sort_values(["n", "method"]).to_csv(path, index=False)
        written["table3"] = str(path)

    # Table 4: learned/exact Merton adjoint validation.
    table4_rows: list[dict[str, Any]] = []
    for value in by_name.values():
        if str(value["tags"].get("table")) != "4":
            continue
        adj = value["adjoint"]
        frame = value["pointwise"]
        table4_rows.append(
            {
                "estimator": value["tags"].get("estimator", value["job"].name),
                "lambda_mean_relative_error": adj.get("lambda_x_mean_relative_error"),
                "lambda_nrmse": adj.get("lambda_x_nrmse"),
                "PXX_nrmse": adj.get("PXX_nrmse"),
                "Z_nrmse": adj.get("Z_x_nrmse"),
                "zeta_rmse": adj.get("zeta_x_rmse_to_exact"),
                "zeta_snr": adj.get("mean_zeta_signal_to_noise"),
                "states": len(frame),
            }
        )
    if table4_rows:
        path = target / "table4_adjoint_validation.csv"
        pd.DataFrame(table4_rows).to_csv(path, index=False)
        written["table4"] = str(path)

    # Tables 5/14: training-budget checkpoint study.
    budget_rows: list[dict[str, Any]] = []
    for value in by_name.values():
        if str(value["tags"].get("table")) not in {"5", "14"}:
            continue
        rec, frame = value["recovery"], value["pointwise"]
        d = int(value["tags"].get("d", 100))
        budget_rows.append(
            {
                "seed": value["tags"].get("seed"),
                "epoch": value["tags"].get("epoch"),
                "dpo_policy_rmse": _rmse_from_norm(frame, "reference_risky_l2_to_exact", d),
                "dpo_kkt_pg": rec.get("reference_kkt_pg_mean"),
                "dpo_H_gap_to_QP": rec.get("full_gain_vs_reference_mean"),
                "barrier_kkt_pg": rec.get("barrier_kkt_pg_mean"),
                "barrier_H_gap_to_QP": rec.get("barrier_gap_to_full_qp_mean"),
                "full_qp_kkt_pg": rec.get("full_kkt_pg_mean"),
                "full_zero_action_l2": rec.get("full_zero_action_l2_mean"),
            }
        )
    if budget_rows:
        raw = pd.DataFrame(budget_rows)
        path = target / "table5_training_budget_by_seed.csv"
        raw.to_csv(path, index=False)
        written["table5_by_seed"] = str(path)
        numeric = [column for column in raw.columns if column not in {"seed", "epoch"}]
        grouped = raw.groupby("epoch")[numeric].agg(["mean", "std"])
        grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
        grouped = grouped.reset_index()
        path2 = target / "table5_training_budget_summary.csv"
        grouped.to_csv(path2, index=False)
        written["table5"] = str(path2)

    # Table 6: Merton consumption-cap.
    cap_rows: list[dict[str, Any]] = []
    for value in by_name.values():
        if str(value["tags"].get("table")) != "6":
            continue
        frame, rec = value["pointwise"], value["recovery"]
        d = int(value["tags"].get("d", 1))
        row_type = value["tags"].get("row_type", "learned")
        cap_rows.append(
            {
                "n": d,
                "row_type": row_type,
                "DPO_u_RMSE": _rmse_from_norm(frame, "reference_risky_l2_to_exact", d),
                "QP_u_RMSE": _rmse_from_norm(frame, "full_risky_l2_to_exact", d),
                "DPO_C_RMSE": _rmse_scalar(frame, "reference_consumption_abs_to_exact"),
                "QP_C_RMSE": _rmse_scalar(frame, "full_consumption_abs_to_exact"),
                "DPO_portfolio_KKT": rec.get("reference_kkt_pg_mean"),
                "QP_portfolio_KKT": rec.get("full_kkt_pg_mean"),
                "DPO_consumption_KKT": rec.get("reference_consumption_kkt_pg_mean"),
                "QP_consumption_KKT": rec.get("full_consumption_kkt_pg_mean"),
                "full_zero_action_l2": rec.get("full_zero_action_l2_mean"),
            }
        )
    if cap_rows:
        path = target / "table6_consumption_cap.csv"
        pd.DataFrame(cap_rows).sort_values(["n", "row_type"]).to_csv(path, index=False)
        written["table6"] = str(path)

    # Table 7: exact affine predictable-return validation.
    exact_affine_rows: list[dict[str, Any]] = []
    for value in by_name.values():
        if str(value["tags"].get("table")) != "7":
            continue
        adj, rec, frame = value["adjoint"], value["recovery"], value["pointwise"]
        exact_affine_rows.append(
            {
                "calibration": value["tags"].get("calibration"),
                "lambda_X_nRMSE": adj.get("lambda_x_nrmse"),
                "PXX_nRMSE": adj.get("PXX_nrmse"),
                "PXY_vs_VXY_nRMSE": adj.get("PXY_vs_VXY_nrmse"),
                "Z_X_nRMSE": adj.get("Z_x_nrmse"),
                "zeta_norm_mean": adj.get("mean_zeta_x_norm"),
                "zeta_SE_mean": adj.get("mean_zeta_se_norm"),
                "zeta_SNR_mean": adj.get("mean_zeta_signal_to_noise"),
                "decoded_policy_coordinatewise_RMSE": _rmse_from_norm(
                    frame, "full_risky_l2_to_exact", 1
                ),
                "zero_policy_coordinatewise_RMSE": _rmse_from_norm(
                    frame, "zero_risky_l2_to_exact", 1
                ),
                "holdout_full_minus_zero": rec.get(
                    "holdout_full_minus_zero_on_full_objective_mean"
                ),
            }
        )
    if exact_affine_rows:
        path = target / "table7_predictable_return.csv"
        pd.DataFrame(exact_affine_rows).to_csv(path, index=False)
        written["table7"] = str(path)

    # Table 8: barrier split audit.
    barrier_rows: list[dict[str, Any]] = []
    for value in by_name.values():
        if str(value["tags"].get("table")) != "8":
            continue
        rec = value["recovery"]
        barrier_rows.append(
            {
                "model": value["tags"].get("model_label"),
                "split": value["tags"].get("split"),
                "portfolio_KKT_PG": rec.get("barrier_kkt_pg_mean"),
                "joint_H_gain_vs_DPO": rec.get("barrier_gain_vs_reference_mean"),
                "joint_H_gap_to_QP": rec.get("barrier_gap_to_full_qp_mean"),
                "converged": rec.get(
                    "barrier_joint_converged_fraction",
                    rec.get("barrier_converged_fraction"),
                ),
                "full_zero_action_l2": rec.get("full_zero_action_l2_mean"),
                "zeta_SNR": rec.get("zeta_signal_to_noise_mean"),
            }
        )
    if barrier_rows:
        path = target / "table8_barrier_audit.csv"
        pd.DataFrame(barrier_rows).to_csv(path, index=False)
        written["table8"] = str(path)

    # Table 9: region-stratified barrier audit, using full-shift QP partition.
    region_rows: list[dict[str, Any]] = []
    region_names = {0: "Upper-active", 1: "Near-switching", 2: "Far-interior"}
    for value in by_name.values():
        if str(value["tags"].get("table")) != "8":
            continue
        frame = value["pointwise"]
        if "switching_region_code" not in frame or "barrier_gain_vs_reference" not in frame:
            continue
        for code, label in region_names.items():
            subset = frame[frame["switching_region_code"] == code]
            if subset.empty:
                continue
            gain = pd.to_numeric(subset["barrier_gain_vs_reference"], errors="coerce")
            gap = pd.to_numeric(subset["barrier_gap_to_full_qp"], errors="coerce")
            region_rows.append(
                {
                    "model": value["tags"].get("model_label"),
                    "split": value["tags"].get("split"),
                    "region": label,
                    "N": len(subset),
                    "mean_H_gain": float(gain.mean()),
                    "minimum_gain": float(gain.min()),
                    "mean_gap_to_QP": float(gap.mean()),
                    "negative_fraction": float((gain < 0).mean()),
                }
            )
    if region_rows:
        raw = pd.DataFrame(region_rows)
        path = target / "table9_switching_by_split.csv"
        raw.to_csv(path, index=False)
        written["table9_by_split"] = str(path)
        # Aggregate split-level summaries with state-count weights.
        weighted = raw.copy()
        for column in ("mean_H_gain", "mean_gap_to_QP", "negative_fraction"):
            weighted[f"weighted_{column}"] = weighted[column] * weighted["N"]
        aggregate = (
            weighted.groupby(["model", "region"], as_index=False)
            .agg(
                N=("N", "sum"),
                weighted_mean_H_gain=("weighted_mean_H_gain", "sum"),
                minimum_gain=("minimum_gain", "min"),
                weighted_mean_gap_to_QP=("weighted_mean_gap_to_QP", "sum"),
                weighted_negative_fraction=("weighted_negative_fraction", "sum"),
            )
        )
        aggregate["mean_H_gain"] = aggregate["weighted_mean_H_gain"] / aggregate["N"]
        aggregate["mean_gap_to_QP"] = (
            aggregate["weighted_mean_gap_to_QP"] / aggregate["N"]
        )
        aggregate["negative_fraction"] = (
            aggregate["weighted_negative_fraction"] / aggregate["N"]
        )
        aggregate = aggregate[[
            "model", "region", "N", "mean_H_gain", "minimum_gain",
            "mean_gap_to_QP", "negative_fraction"
        ]]
        path2 = target / "table9_switching_aggregate.csv"
        aggregate.to_csv(path2, index=False)
        written["table9"] = str(path2)

    # Appendix Table 12: constrained LKO full-shift recovery and shift audit.
    lko_rows: list[dict[str, Any]] = []
    for value in by_name.values():
        if str(value["tags"].get("table")) != "12":
            continue
        rec = value["recovery"]
        lko_rows.append(
            {
                "n": value["tags"].get("d"),
                "DPO_KKT": rec.get("reference_kkt_pg_mean"),
                "B_PGDPO_KKT": rec.get("barrier_kkt_pg_mean"),
                "QP_PGDPO_KKT": rec.get("full_kkt_pg_mean"),
                "B_PGDPO_gain": rec.get("barrier_gain_vs_reference_mean"),
                "QP_PGDPO_gain": rec.get("full_gain_vs_reference_mean"),
                "zeta_over_Z": value["adjoint"].get("mean_zeta_to_Z_ratio"),
                "zeta_SNR": value["adjoint"].get("mean_zeta_signal_to_noise"),
                "zeta_replication_l2": rec.get("zeta_replication_l2_mean"),
                "full_zero_action_l2": rec.get("full_zero_action_l2_mean"),
                "holdout_full_minus_zero": rec.get(
                    "holdout_full_minus_zero_on_full_objective_mean"
                ),
            }
        )
    if lko_rows:
        path = target / "table12_lko_full_shift.csv"
        pd.DataFrame(lko_rows).sort_values("n").to_csv(path, index=False)
        written["table12"] = str(path)

    # Appendix Table 13: decoder initialization audit.
    initialization_rows: list[dict[str, Any]] = []
    for value in by_name.values():
        if str(value["tags"].get("table")) != "13":
            continue
        audit = value["initialization"]
        if not audit:
            continue
        initialization_rows.append(
            {
                "model": value["tags"].get("model_label"),
                "N": audit.get("states"),
                "exact_QP_uses_initialization": audit.get(
                    "exact_qp_uses_initialization"
                ),
                "exact_max_difference": 0.0,
                "PGD_mean_difference": audit.get("mean_difference"),
                "PGD_max_difference": audit.get("maximum_difference"),
                "PGD_iterations": audit.get("iterations"),
            }
        )
    if initialization_rows:
        path = target / "table13_initialization.csv"
        pd.DataFrame(initialization_rows).to_csv(path, index=False)
        written["table13"] = str(path)

    # Appendix graph-ablation summary.
    graph_rows: list[dict[str, Any]] = []
    for value in by_name.values():
        if value["job"].stage != "graph_compare" or not value["graph"]:
            continue
        graph_rows.append(
            {
                "model": value["tags"].get("model_label", value["job"].name),
                **value["graph"],
            }
        )
    if graph_rows:
        path = target / "appendix_graph_ablation.csv"
        pd.DataFrame(graph_rows).to_csv(path, index=False)
        written["graph_ablation"] = str(path)

    # Appendix Table 15: barrier epsilon sweep.
    epsilon_rows: list[dict[str, Any]] = []
    for value in by_name.values():
        if str(value["tags"].get("table")) != "15":
            continue
        rec = value["recovery"]
        epsilon_rows.append(
            {
                "model": value["tags"].get("model_label"),
                "epsilon": value["tags"].get("epsilon"),
                "mean_gain": rec.get("barrier_gain_vs_reference_mean"),
                "minimum_gain": rec.get("barrier_gain_vs_reference_min"),
                "negative_share": None,
                "gap_to_QP": rec.get("barrier_gap_to_full_qp_mean"),
                "converged": rec.get(
                    "barrier_joint_converged_fraction",
                    rec.get("barrier_converged_fraction"),
                ),
            }
        )
        frame = value["pointwise"]
        if not frame.empty and "barrier_gain_vs_reference" in frame:
            epsilon_rows[-1]["negative_share"] = float(
                (pd.to_numeric(frame["barrier_gain_vs_reference"], errors="coerce") < 0).mean()
            )
    if epsilon_rows:
        path = target / "table15_barrier_epsilon.csv"
        pd.DataFrame(epsilon_rows).to_csv(path, index=False)
        written["table15"] = str(path)

    # Estimator ablation: automatically available from each exact job.
    ablation_rows: list[dict[str, Any]] = []
    for value in by_name.values():
        if not bool(value["tags"].get("estimator_ablation", False)):
            continue
        adj = value["adjoint"]
        for estimator in (
            "raw_moment",
            "centered_moment",
            "control_variate",
            "ols_control_variate",
        ):
            ablation_rows.append(
                {
                    "model": value["tags"].get("model_label", value["job"].name),
                    "estimator": estimator,
                    "Z_X_nRMSE": adj.get(f"{estimator}_Z_x_nrmse"),
                    "zeta_RMSE": adj.get(f"{estimator}_zeta_x_rmse_to_exact"),
                    "zeta_norm": adj.get(f"{estimator}_mean_zeta_x_norm"),
                    "zeta_SE": adj.get(f"{estimator}_mean_zeta_se_norm"),
                }
            )
    if ablation_rows:
        path = target / "appendix_z_estimator_ablation.csv"
        pd.DataFrame(ablation_rows).to_csv(path, index=False)
        written["z_estimator_ablation"] = str(path)

    # Appendix Table 16 internal PGDPO rows (external hPINN values are merged manually).
    table16_rows: list[dict[str, Any]] = []
    for value in by_name.values():
        include = str(value["tags"].get("appendix_table")) == "16"
        include = include or (
            str(value["tags"].get("table")) == "12"
            and int(value["tags"].get("d", 0)) == 100
        )
        if not include or not value["recovery"]:
            continue
        rec = value["recovery"]
        table16_rows.append(
            {
                "benchmark": value["tags"].get("model_label"),
                "d": value["tags"].get("d"),
                "dY": value["tags"].get("k"),
                "DPO_KKT": rec.get("reference_kkt_pg_mean"),
                "B_PGDPO_KKT": rec.get("barrier_kkt_pg_mean"),
                "QP_PGDPO_KKT": rec.get("full_kkt_pg_mean"),
                "QP_PGDPO_gain": rec.get("full_gain_vs_reference_mean"),
                "full_zero_action_l2": rec.get("full_zero_action_l2_mean"),
            }
        )
    if table16_rows:
        path = target / "table16_internal_pgdpo.csv"
        pd.DataFrame(table16_rows).to_csv(path, index=False)
        written["table16_internal"] = str(path)

    # Timing: sum newly measured stages; historical DPO warm-up can be merged later.
    timing_rows: list[dict[str, Any]] = []
    for value in by_name.values():
        if str(value["tags"].get("table")) != "17":
            continue
        timing = value["timing"]
        harvest = float(timing.get("harvest_seconds", 0.0))
        holdout = float(timing.get("holdout_harvest_seconds", 0.0))
        recovery = float(timing.get("recovery_seconds", 0.0))
        revalidation = harvest + holdout + recovery
        timing_rows.append(
            {
                "dY": value["tags"].get("k"),
                "historical_DPO_warmup_minutes": value["tags"].get(
                    "historical_dpo_minutes"
                ),
                "OL_BPTT_Z_projection_QP_seconds": harvest + recovery,
                "holdout_seconds": holdout,
                "total_revalidation_seconds": revalidation,
                "total_revalidation_minutes": revalidation / 60.0,
                "QP_KKT": value["recovery"].get("full_kkt_pg_mean"),
            }
        )
    if timing_rows:
        path = target / "table17_timing.csv"
        pd.DataFrame(timing_rows).sort_values("dY").to_csv(path, index=False)
        written["table17"] = str(path)

    (target / "collection_manifest.json").write_text(
        json.dumps(written, indent=2, sort_keys=True), encoding="utf-8"
    )
    return written
