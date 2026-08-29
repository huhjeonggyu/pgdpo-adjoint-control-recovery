from __future__ import annotations

import torch

from mf_revision.models.market import affine_joint_market, legacy_merton_cap_market


def test_merton_cap_loading_matches_reported_covariance() -> None:
    market = legacy_merton_cap_market(
        10,
        gamma=2.0,
        seed=42,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    implied = market["loading"] @ market["loading"].T
    torch.testing.assert_close(market["covariance"], implied, rtol=0.0, atol=1e-12)


def test_affine_loading_matches_covariance_and_cross_covariance() -> None:
    market = affine_joint_market(
        10,
        3,
        seed=11,
        device=torch.device("cpu"),
        dtype=torch.float64,
        mode="conservative",
    )
    asset_loading = market["asset_loading"]
    factor_loading = market["factor_loading"]
    torch.testing.assert_close(
        market["covariance"],
        asset_loading @ asset_loading.T,
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        market["cross_covariance"],
        asset_loading @ factor_loading.T,
        rtol=0.0,
        atol=1e-12,
    )
