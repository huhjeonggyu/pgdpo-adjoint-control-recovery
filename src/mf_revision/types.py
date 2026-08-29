"""Shared typed result containers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import torch

Tensor = torch.Tensor


@dataclass(slots=True)
class AdjointEstimate:
    """Adapted adjoint tuple at diagnostic states.

    For ``B`` states, state dimension ``n``, and Brownian dimension ``q``:

    - ``lambda_`` has shape ``[B,n]``;
    - ``p`` has shape ``[B,n,n]``;
    - ``z`` and ``zeta`` have shape ``[B,n,q]``;
    - ``sigma_ref`` has shape ``[B,n,q]``.
    """

    states: Tensor
    tau: Tensor
    lambda_: Tensor
    p: Tensor
    z: Tensor
    zeta: Tensor
    sigma_ref: Tensor
    reference_control: Tensor
    lambda_se: Tensor | None = None
    p_se: Tensor | None = None
    z_se: Tensor | None = None
    zeta_se: Tensor | None = None
    projection_variants: dict[str, Tensor] = field(default_factory=dict)
    projection_se_variants: dict[str, Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def lambda_x(self) -> Tensor:
        return self.lambda_[:, 0:1]

    @property
    def p_xx(self) -> Tensor:
        return self.p[:, 0:1, 0:1]

    @property
    def p_xrow(self) -> Tensor:
        return self.p[:, 0, :]

    @property
    def z_x(self) -> Tensor:
        return self.z[:, 0, :]

    @property
    def zeta_x(self) -> Tensor:
        return self.zeta[:, 0, :]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({name: getattr(self, name) for name in self.__dataclass_fields__}, target)

    @classmethod
    def load(
        cls, path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> "AdjointEstimate":
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid adjoint file: {path}")
        return cls(**payload)


@dataclass(slots=True)
class RecoveryResult:
    full_control: Tensor
    zero_shift_control: Tensor
    barrier_control: Tensor | None
    g_full: Tensor
    g_zero: Tensor
    q: Tensor
    diagnostics: dict[str, Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({name: getattr(self, name) for name in self.__dataclass_fields__}, target)

    @classmethod
    def load(
        cls, path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> "RecoveryResult":
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid recovery file: {path}")
        return cls(**payload)
