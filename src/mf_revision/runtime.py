"""Runtime, seed, dtype, and provenance helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import os
import random
import subprocess

import numpy as np
import torch


def resolve_device(name: str) -> torch.device:
    value = str(name).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def resolve_dtype(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"float64", "double", "fp64"}:
        return torch.float64
    if value in {"float32", "single", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name!r}")


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(path: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def environment_card(project_root: str | Path) -> dict[str, Any]:
    return {
        "python": os.sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "git_commit": git_commit(project_root),
    }
