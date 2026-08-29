"""Discovery of the historical paper checkpoints in ``legacy_mf_revision``.

The legacy archive contains many revision-stage copies of the same trained
policies.  This module gives stable semantic aliases to the checkpoints used by
the manuscript tables.  No legacy Python module is imported; only state_dict
files and optional market snapshots are reused.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import os


@dataclass(frozen=True, slots=True)
class LegacyAsset:
    alias: str
    kind: str
    patterns: tuple[str, ...]
    description: str


_ASSETS: tuple[LegacyAsset, ...] = (
    LegacyAsset(
        "merton_short_d10_e500",
        "checkpoint",
        (
            "PGDPO_TORCH-main/revision_stage2_d10_e500/mt_nd_short/base/**/policy_stage1.pt",
            "stage4_test/PGDPO_TORCH-main/revision_stage4_d10_e500/mt_nd_short/base/**/policy_stage1.pt",
        ),
        "Borrowing-allowed no-short-sale Merton, d=10, 500 epochs.",
    ),
    LegacyAsset(
        "merton_short_d100_e500",
        "checkpoint",
        (
            "PGDPO_TORCH-main/revision_stage2_d100_e500/mt_nd_short/base/**/policy_stage1.pt",
            "stage4_test/PGDPO_TORCH-main/revision_stage4_d100_e500/mt_nd_short/base/**/policy_stage1.pt",
        ),
        "Borrowing-allowed no-short-sale Merton, d=100, 500 epochs.",
    ),
    LegacyAsset(
        "merton_cap_d2_e500",
        "checkpoint",
        (
            "stage5b_test/PGDPO_TORCH-main/revision_stage5b_mtcap_d2_e500/mt_nd_cap_short/base/**/policy_stage1.pt",
        ),
        "Merton proportional-consumption-cap checkpoint, d=2.",
    ),
    LegacyAsset(
        "merton_cap_d10_e500",
        "checkpoint",
        (
            "stage5b_test/PGDPO_TORCH-main/revision_stage5b_mtcap_d10_e500/mt_nd_cap_short/base/**/policy_stage1.pt",
        ),
        "Merton proportional-consumption-cap checkpoint, d=10.",
    ),
    LegacyAsset(
        "merton_cap_d100_e500",
        "checkpoint",
        (
            "stage5b_test/PGDPO_TORCH-main/revision_stage5b_mtcap_d100_light/mt_nd_cap_short/base/**/policy_stage1.pt",
        ),
        "Merton proportional-consumption-cap checkpoint, d=100.",
    ),
    LegacyAsset(
        "lko_short_d10_e500",
        "checkpoint",
        (
            "stage5_test/PGDPO_TORCH-main/revision_stage5_d10_k1_e500/liu_ko_nd_short/base/**/policy_stage1.pt",
        ),
        "Constrained Liu--Kim--Omberg checkpoint, d=10, one factor.",
    ),
    LegacyAsset(
        "lko_short_d50_e500",
        "checkpoint",
        (
            "stage5_test/PGDPO_TORCH-main/revision_stage5_d50_k1_e500/liu_ko_nd_short/base/**/policy_stage1.pt",
        ),
        "Constrained Liu--Kim--Omberg checkpoint, d=50, one factor.",
    ),
    LegacyAsset(
        "lko_short_d100_e500",
        "checkpoint",
        (
            "stage5_test/PGDPO_TORCH-main/revision_stage5_d100_k1_light/liu_ko_nd_short/base/**/policy_stage1.pt",
        ),
        "Constrained Liu--Kim--Omberg checkpoint, d=100, one factor.",
    ),
    LegacyAsset(
        "lko_cap_d10_e500",
        "checkpoint",
        (
            "stage5b_test/PGDPO_TORCH-main/revision_stage5b_liucap_d10_e500/liu_ko_nd_cap/base/**/policy_stage1.pt",
        ),
        "Constrained factor consumption-cap checkpoint, d=10.",
    ),
    LegacyAsset(
        "lko_cap_d50_e500",
        "checkpoint",
        (
            "stage5b_test/PGDPO_TORCH-main/revision_stage5b_liucap_d50_e500/liu_ko_nd_cap/base/**/policy_stage1.pt",
        ),
        "Constrained factor consumption-cap checkpoint, d=50.",
    ),
    LegacyAsset(
        "lko_cap_d100_e500",
        "checkpoint",
        (
            "stage5b_test/PGDPO_TORCH-main/revision_stage5b_liucap_d100_light/liu_ko_nd_cap/base/**/policy_stage1.pt",
        ),
        "Constrained factor consumption-cap checkpoint, d=100.",
    ),
    LegacyAsset(
        "merton_budget_market",
        "market_snapshot",
        (
            "stage11_test/PGDPO_TORCH-main/revision_stage15_ol_full_gpu7/market_snapshot.pt",
            "stage11_test/PGDPO_TORCH-main/revision_stage15_training_budget/market_snapshot.pt",
        ),
        "Common d=100 Merton market used by the three-seed budget study.",
    ),
    LegacyAsset(
        "affine_old_d100_k1",
        "checkpoint",
        (
            "stage11_test/PGDPO_TORCH-main/revision_stage11_old_affine_multifactor_timing/liu_ko_nd_short/base/d100_k1/**/policy_stage1.pt",
        ),
        "Supplementary old-affine factor checkpoint, d=100, k=1.",
    ),
    LegacyAsset(
        "affine_old_d100_k3",
        "checkpoint",
        (
            "stage11_test/PGDPO_TORCH-main/revision_stage11_old_affine_multifactor_timing/liu_ko_nd_short/base/d100_k3/**/policy_stage1.pt",
        ),
        "Supplementary old-affine factor checkpoint, d=100, k=3.",
    ),
    LegacyAsset(
        "affine_old_d100_k5",
        "checkpoint",
        (
            "stage11_test/PGDPO_TORCH-main/revision_stage11_old_affine_multifactor_timing/liu_ko_nd_short/base/d100_k5/**/policy_stage1.pt",
        ),
        "Supplementary old-affine factor checkpoint, d=100, k=5.",
    ),
)


def _all_assets() -> Iterable[LegacyAsset]:
    yield from _ASSETS
    for seed in (1, 2, 3):
        for epoch in (250, 500, 1000):
            yield LegacyAsset(
                f"merton_budget_seed{seed}_e{epoch}",
                "checkpoint",
                (
                    "stage11_test/PGDPO_TORCH-main/"
                    f"revision_stage15_ol_full_gpu7/checkpoints/seed_{seed:04d}/"
                    f"checkpoint_e{epoch:04d}.pt",
                    "stage11_test/PGDPO_TORCH-main/"
                    f"revision_stage15_training_budget/checkpoints/seed_{seed:04d}/"
                    f"checkpoint_e{epoch:04d}.pt",
                ),
                f"Merton budget study, training seed {seed}, epoch {epoch}.",
            )


def _choose(matches: list[Path]) -> Path:
    """Choose a deterministic candidate, preferring the newest timestamp path."""
    if not matches:
        raise ValueError("No candidates")
    # Multiple d=100 light runs can exist.  The later path name is the final
    # replacement and is preferred.  Sorting also makes discovery reproducible.
    return sorted((path.resolve() for path in matches), key=lambda p: str(p))[-1]


def discover_legacy_assets(root: str | Path, *, strict: bool = False) -> dict[str, str]:
    base = Path(root).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Legacy root not found: {base}")
    discovered: dict[str, str] = {}
    missing: list[str] = []
    for asset in _all_assets():
        selected: Path | None = None
        for pattern in asset.patterns:
            matches = list(base.glob(pattern))
            if matches:
                selected = _choose(matches)
                break
        if selected is None:
            missing.append(asset.alias)
        else:
            discovered[asset.alias] = str(selected)
    if strict and missing:
        raise FileNotFoundError(
            "Missing required legacy assets: " + ", ".join(sorted(missing))
        )
    return discovered


def write_legacy_catalog(
    root: str | Path,
    output: str | Path,
    *,
    strict: bool = False,
) -> dict[str, str]:
    values = discover_legacy_assets(root, strict=strict)
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(values, indent=2, sort_keys=True), encoding="utf-8")
    return values


def load_legacy_catalog(path: str | Path) -> dict[str, str]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Legacy catalog must be a JSON mapping")
    values: dict[str, str] = {}
    for key, value in payload.items():
        asset = Path(os.path.expandvars(str(value))).expanduser()
        if not asset.is_absolute():
            asset = source.parent / asset
        values[str(key)] = str(asset.resolve())
    return values
