#!/usr/bin/env python3
r"""Post-hoc zero-shift ablation for the exact one-factor predictable-return runs.

No retraining and no OL-BPTT rerun are required.  The script reads the saved
adjoint/recovery tensors from the two Table-7 runs and reports

1) the raw P^{XY} vs V_{XY} diagnostic,
2) the identity-consistent reconstruction
       e_X^T D^2 V = e_X^T P + zeta^X sigma_S^\dagger,
3) analytical-policy RMSE for the saved full-shift and zero-shift decoders,
4) shift SNR and conditioning diagnostics.

Run from the repository root, e.g.

    python zero_shift_vxy_ablation.py

or

    python zero_shift_vxy_ablation.py --root /path/to/pgdpo-adjoint-control-recovery

Outputs are written to <root>/paper/zero_shift_vxy_ablation/ by default.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

RUNS = ("t7_affine_canonical", "t7_affine_hedging")


def nrmse(estimate: torch.Tensor, target: torch.Tensor) -> float:
    numerator = torch.mean((estimate - target).square()).sqrt()
    denominator = torch.mean(target.square()).sqrt().clamp_min(1e-14)
    return float((numerator / denominator).cpu())


def rmse(estimate: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((estimate - target).square()).sqrt().cpu())


def reconstruct_hessian_wealth_row(
    p_xrow: torch.Tensor,
    zeta_x: torch.Tensor,
    sigma_state: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct e_X^T D^2 V from zeta^X=(e_X^T D^2V-e_X^T P)sigma.

    We solve delta @ sigma = zeta, state by state.  ``pinv`` keeps the formula
    valid for a merely full-row-rank diffusion and makes rank/conditioning
    diagnostics transparent.
    """
    sigma_pinv = torch.linalg.pinv(sigma_state)
    correction = torch.einsum("bq,bqs->bs", zeta_x, sigma_pinv)
    return p_xrow + correction


def load_model(root: Path, run_dir: Path):
    # Make the repository importable even when it has not been pip-installed.
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from mf_revision.models.factory import build_model

    config_path = run_dir / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    model = build_model(model_cfg, device=torch.device("cpu"), dtype=torch.float64)
    return model, config


def analyze_run(root: Path, run_name: str, output_dir: Path) -> dict[str, float | str]:
    run_dir = root / "runs_paper_full_shift" / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")

    adj = torch.load(run_dir / "adjoints.pt", map_location="cpu", weights_only=False)
    recovery = torch.load(run_dir / "recovery.pt", map_location="cpu", weights_only=False)
    model, _ = load_model(root, run_dir)

    states = adj["states"].to(torch.float64)
    tau = adj["tau"].to(torch.float64)
    p = adj["p"].to(torch.float64)
    zeta = adj["zeta"].to(torch.float64)
    sigma_ref = adj["sigma_ref"].to(torch.float64)
    zeta_se = adj.get("zeta_se")
    if zeta_se is not None:
        zeta_se = zeta_se.to(torch.float64)

    exact = model.exact_adjoint_fields(states, tau)
    vxy_true = exact["VXY"].to(torch.float64)              # [B,1]
    hessian_true = exact["value_hessian_wealth_row"].to(torch.float64)
    policy_true = exact["control"].to(torch.float64)

    p_xrow = p[:, 0, :]                                     # [B,2]
    pxy = p_xrow[:, 1:2]
    zeta_x = zeta[:, 0, :]                                  # [B,2]

    hessian_recon = reconstruct_hessian_wealth_row(p_xrow, zeta_x, sigma_ref)
    vxy_recon = hessian_recon[:, 1:2]
    vxx_recon = hessian_recon[:, 0:1]

    # A literal zero-shift Hessian shortcut corresponds to D^2V wealth row ~= P wealth row.
    vxy_zero = pxy

    full_policy = recovery["full_control"].to(torch.float64)
    zero_policy = recovery["zero_shift_control"].to(torch.float64)

    singular_values = torch.linalg.svdvals(sigma_ref)
    sigma_min = singular_values[:, -1]
    sigma_max = singular_values[:, 0]
    sigma_cond = sigma_max / sigma_min.clamp_min(1e-14)
    sigma_rank = torch.linalg.matrix_rank(sigma_ref)

    if zeta_se is not None:
        zeta_se_x = zeta_se[:, 0, :]
        zeta_snr_state = torch.linalg.norm(zeta_x, dim=-1) / torch.linalg.norm(
            zeta_se_x, dim=-1
        ).clamp_min(1e-14)
        mean_zeta_snr = float(zeta_snr_state.mean().cpu())
    else:
        zeta_snr_state = torch.full((states.shape[0],), float("nan"), dtype=torch.float64)
        mean_zeta_snr = float("nan")

    summary: dict[str, float | str] = {
        "run": run_name,
        "n_states": int(states.shape[0]),
        "PXY_vs_VXY_nRMSE": nrmse(vxy_zero, vxy_true),
        "reconstructed_VXY_nRMSE": nrmse(vxy_recon, vxy_true),
        "reconstructed_VXX_nRMSE": nrmse(vxx_recon, exact["VXX"].to(torch.float64)),
        "reconstructed_hessian_row_nRMSE": nrmse(hessian_recon, hessian_true),
        "full_shift_policy_RMSE": rmse(full_policy, policy_true),
        "zero_shift_policy_RMSE": rmse(zero_policy, policy_true),
        "full_vs_zero_policy_RMSE": rmse(full_policy, zero_policy),
        "mean_zeta_SNR": mean_zeta_snr,
        "mean_zeta_norm": float(torch.linalg.norm(zeta_x, dim=-1).mean().cpu()),
        "median_sigma_condition": float(sigma_cond.median().cpu()),
        "max_sigma_condition": float(sigma_cond.max().cpu()),
        "min_sigma_singular_value": float(sigma_min.min().cpu()),
        "min_sigma_rank": int(sigma_rank.min().cpu()),
    }

    frame = pd.DataFrame(
        {
            "wealth": states[:, 0].numpy(),
            "factor": states[:, 1].numpy(),
            "tau": tau[:, 0].numpy(),
            "VXY_true": vxy_true[:, 0].numpy(),
            "PXY_zero_shift": vxy_zero[:, 0].numpy(),
            "VXY_reconstructed_full_shift": vxy_recon[:, 0].numpy(),
            "VXX_true": exact["VXX"][:, 0].detach().cpu().numpy(),
            "PXX": p_xrow[:, 0].numpy(),
            "VXX_reconstructed_full_shift": vxx_recon[:, 0].numpy(),
            "zeta_X_0": zeta_x[:, 0].numpy(),
            "zeta_X_1": zeta_x[:, 1].numpy(),
            "zeta_SNR_state": zeta_snr_state.numpy(),
            "sigma_min_sv": sigma_min.numpy(),
            "sigma_condition": sigma_cond.numpy(),
            "policy_true": policy_true[:, 0].detach().cpu().numpy(),
            "policy_full_shift": full_policy[:, 0].numpy(),
            "policy_zero_shift": zero_policy[:, 0].numpy(),
        }
    )
    frame["PXY_abs_error"] = (frame["PXY_zero_shift"] - frame["VXY_true"]).abs()
    frame["VXY_recon_abs_error"] = (
        frame["VXY_reconstructed_full_shift"] - frame["VXY_true"]
    ).abs()
    frame["policy_full_abs_error"] = (
        frame["policy_full_shift"] - frame["policy_true"]
    ).abs()
    frame["policy_zero_abs_error"] = (
        frame["policy_zero_shift"] - frame["policy_true"]
    ).abs()

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{run_name}_pointwise.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="repository root (default: current directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory (default: <root>/paper/zero_shift_vxy_ablation)",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output_dir = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "paper" / "zero_shift_vxy_ablation"
    )

    summaries = [analyze_run(root, run_name, output_dir) for run_name in RUNS]
    table = pd.DataFrame(summaries)
    table.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )

    columns = [
        "run",
        "PXY_vs_VXY_nRMSE",
        "reconstructed_VXY_nRMSE",
        "full_shift_policy_RMSE",
        "zero_shift_policy_RMSE",
        "full_vs_zero_policy_RMSE",
        "mean_zeta_SNR",
        "median_sigma_condition",
    ]
    print("\nZero-shift / full-shift predictable-return ablation")
    print(table[columns].to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved: {output_dir / 'summary.csv'}")
    print(f"Saved pointwise files under: {output_dir}")


if __name__ == "__main__":
    main()
