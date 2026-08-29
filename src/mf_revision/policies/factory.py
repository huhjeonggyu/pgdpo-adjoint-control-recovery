"""Policy and chart factories, including legacy checkpoint compatibility."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import torch

from mf_revision.models.base import PortfolioModel
from .charts import ControlChart
from .legacy import LegacyFactorCapPolicy, LegacyMertonCapPolicy, LegacySingleNetworkPolicy
from .neural import NeuralPolicy


def build_chart(model: PortfolioModel) -> ControlChart:
    return ControlChart(
        model.dims.risky,
        constraint=model.constraint,
        leverage_cap=model.leverage_cap,
        consumption_rate_min=model.consumption_rate_min,
        consumption_rate_max=model.consumption_rate_max if model.has_consumption else None,
    )


def build_policy(
    model: PortfolioModel,
    config: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
):
    kind = str(config.get("kind", "mlp")).lower()
    if kind == "analytic":
        return model.analytical_policy().to(device=device, dtype=dtype)
    chart = build_chart(model)
    architecture = str(config.get("architecture", "modern")).lower().replace("-", "_")
    hidden = config.get("hidden", [200, 200])
    activation = config.get("activation", "leaky_relu")
    if architecture in {"modern", "default"}:
        policy = NeuralPolicy(
            model.dims.state,
            model.T,
            chart,
            hidden=hidden,
            activation=activation,
            feature_mode=config.get("feature_mode", "raw"),
        )
    elif architecture in {"legacy_single", "legacy_short", "legacy"}:
        # Merton-short historically used PyTorch default initialization; LKO used Xavier.
        xavier = bool(config.get("xavier", model.dims.factor > 0 or model.constraint == "simplex"))
        policy = LegacySingleNetworkPolicy(
            model.dims.state,
            chart,
            hidden=hidden,
            activation=activation,
            xavier=xavier,
        )
    elif architecture in {"legacy_merton_cap", "legacy_mt_cap"}:
        policy = LegacyMertonCapPolicy(
            model.dims.state,
            chart,
            hidden=hidden,
            activation=activation,
        )
    elif architecture in {"legacy_factor_cap", "legacy_lko_cap"}:
        policy = LegacyFactorCapPolicy(
            model.dims.state,
            chart,
            hidden=hidden,
            activation=activation,
        )
    else:
        raise ValueError(f"Unknown policy architecture: {architecture}")
    return policy.to(device=device, dtype=dtype)


def save_policy(path: str | Path, policy: torch.nn.Module, *, metadata: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "metadata": metadata}, target)


def _extract_state_dict(payload: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a mapping")
    for key in ("state_dict", "policy_state_dict", "model_state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            metadata = {name: value for name, value in payload.items() if name != key}
            return candidate, metadata
    # Some research checkpoints are a raw state_dict.
    if payload and all(isinstance(key, str) for key in payload) and all(
        torch.is_tensor(value) for value in payload.values()
    ):
        return payload, {}
    raise ValueError("No state_dict found in checkpoint")


def load_policy_checkpoint(
    path: str | Path,
    policy: torch.nn.Module,
    *,
    map_location: torch.device,
    strict: bool = True,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    state_dict, metadata = _extract_state_dict(payload)
    incompatible = policy.load_state_dict(state_dict, strict=bool(strict))
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        metadata = {
            **metadata,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
    return dict(metadata)
