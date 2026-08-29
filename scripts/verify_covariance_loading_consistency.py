#!/usr/bin/env python3
"""Verify simulator Brownian loadings are consistent with reported covariances.

Run from the repository root. The check is also exercised by the test suite.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch


def _relative_fro(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(torch.linalg.norm(a, ord="fro").clamp_min(1e-30))
    return float(torch.linalg.norm(a - b, ord="fro")) / denom


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tol", type=float, default=1e-12)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    from mf_revision.models.market import affine_joint_market, legacy_merton_cap_market

    device = torch.device("cpu")
    dtype = torch.float64
    rows: list[tuple[str, float, float]] = []

    for d in (2, 10, 100):
        market = legacy_merton_cap_market(
            d, gamma=2.0, seed=42, device=device, dtype=dtype
        )
        sigma = market["covariance"]
        implied = market["loading"] @ market["loading"].T
        rows.append((f"merton_cap d={d}", _relative_fro(sigma, implied), _max_abs(sigma, implied)))

    for mode in ("conservative", "old_affine"):
        for d, k in ((10, 1), (50, 1), (100, 1), (100, 3), (100, 5)):
            market = affine_joint_market(
                d, k, seed=11, device=device, dtype=dtype, mode=mode
            )
            sigma = market["covariance"]
            asset_loading = market["asset_loading"]
            factor_loading = market["factor_loading"]
            implied_sigma = asset_loading @ asset_loading.T
            implied_cross = asset_loading @ factor_loading.T
            rel_sigma = _relative_fro(sigma, implied_sigma)
            max_sigma = _max_abs(sigma, implied_sigma)
            rel_cross = _relative_fro(market["cross_covariance"], implied_cross)
            max_cross = _max_abs(market["cross_covariance"], implied_cross)
            rows.append((f"affine[{mode}] d={d} k={k} asset", rel_sigma, max_sigma))
            rows.append((f"affine[{mode}] d={d} k={k} cross", rel_cross, max_cross))

    print(f"{'case':46s} {'rel_fro':>14s} {'max_abs':>14s}")
    print("-" * 78)
    failures = 0
    for name, rel, mx in rows:
        print(f"{name:46s} {rel:14.6e} {mx:14.6e}")
        if not (math.isfinite(rel) and math.isfinite(mx) and rel <= args.tol and mx <= args.tol):
            failures += 1

    if failures:
        print(f"\nFAIL: {failures} consistency checks exceed tol={args.tol:g}")
        return 1
    print(f"\nPASS: all loading/covariance identities hold within tol={args.tol:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
