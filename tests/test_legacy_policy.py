from __future__ import annotations

from pathlib import Path
import torch

from mf_revision.models.factory import build_model
from mf_revision.policies.factory import build_policy, load_policy_checkpoint


def test_legacy_merton_cap_checkpoint_roundtrip(tmp_path: Path) -> None:
    device = torch.device("cpu")
    dtype = torch.float64
    model = build_model(
        {
            "name": "merton",
            "d": 2,
            "T": 1.0,
            "n_steps": 4,
            "r": 0.03,
            "rho": 0.1,
            "gamma": 2.0,
            "constraint": "simplex",
            "leverage_cap": 1.0,
            "consumption": True,
            "consumption_rate_min": 1e-8,
            "consumption_rate_max": 0.7,
            "market_mode": "legacy_cap",
            "market_seed": 42,
        },
        device=device,
        dtype=dtype,
    )
    config = {
        "kind": "mlp",
        "architecture": "legacy_merton_cap",
        "hidden": [8, 8],
    }
    first = build_policy(model, config, device=device, dtype=dtype)
    checkpoint = tmp_path / "legacy.pt"
    torch.save(first.state_dict(), checkpoint)
    second = build_policy(model, config, device=device, dtype=dtype)
    load_policy_checkpoint(checkpoint, second, map_location=device, strict=True)
    state = torch.tensor([[1.0]], dtype=dtype)
    tau = torch.tensor([[0.5]], dtype=dtype)
    assert torch.equal(first(state, tau), second(state, tau))
