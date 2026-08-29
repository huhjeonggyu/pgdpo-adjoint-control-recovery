"""Model factory."""
from __future__ import annotations

from typing import Any
import torch

from .base import PortfolioModel
from .merton import MertonModel
from .affine_factor import AffineFactorModel


def build_model(
    config: dict[str, Any], *, device: torch.device, dtype: torch.dtype
) -> PortfolioModel:
    name = str(config.get("name", "")).lower()
    if name == "merton":
        return MertonModel(config, device=device, dtype=dtype)
    if name in {"affine_factor", "liu_ko", "predictable_return"}:
        return AffineFactorModel(config, device=device, dtype=dtype)
    raise ValueError(f"Unknown model.name={name!r}")
