"""YAML experiment configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import copy
import hashlib
import json
import os

import yaml


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    seed: int = 12345
    device: str = "auto"
    dtype: str = "float64"
    output_root: str = "runs"
    model: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    recovery: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, source_path: str | None = None
    ) -> "ExperimentConfig":
        data = _expand_environment(copy.deepcopy(dict(value)))
        data["source_path"] = source_path
        config = cls(**data)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.name:
            raise ValueError("name is required")
        if not self.model.get("name"):
            raise ValueError("model.name is required")
        if str(self.policy.get("kind", "mlp")).lower() not in {"mlp", "analytic"}:
            raise ValueError("policy.kind must be mlp or analytic")
        if str(self.evaluation.get("graph_mode", "ol")).lower() not in {"ol", "cl"}:
            raise ValueError("evaluation.graph_mode must be ol or cl")
        state_mode = str(self.evaluation.get("state_mode", "iid")).lower().replace("-", "_")
        if state_mode not in {
            "iid", "train", "default", "wealth_edge", "wealth",
            "time_edge", "horizon_edge", "factor_edge", "factor",
            "stress", "joint_stress", "custom",
        }:
            raise ValueError("evaluation.state_mode is invalid")
        z_estimator = str(
            self.evaluation.get("z_estimator", "raw_moment")
        ).lower().replace("-", "_")
        z_aliases = {
            "raw": "raw_moment",
            "legacy": "raw_moment",
            "centered": "centered_moment",
            "cv": "control_variate",
            "ols": "ols_control_variate",
            "nested": "nested_crn_ols",
            "nested_ols": "nested_crn_ols",
            "crn_ols": "nested_crn_ols",
        }
        z_estimator = z_aliases.get(z_estimator, z_estimator)
        if z_estimator not in {
            "raw_moment",
            "centered_moment",
            "control_variate",
            "ols_control_variate",
            "nested_crn_ols",
        }:
            raise ValueError("evaluation.z_estimator is invalid")
        outer = int(
            self.evaluation.get(
                "z_outer_paths", self.evaluation.get("continuations", 8192)
            )
        )
        inner_default = 8 if z_estimator == "nested_crn_ols" else 1
        inner = int(self.evaluation.get("z_inner_paths", inner_default))
        pair_batch = int(self.evaluation.get("z_outer_pair_batch", 32))
        if outer <= 0 or outer % 2:
            raise ValueError("evaluation.z_outer_paths must be positive and even")
        if inner <= 0 or (inner > 1 and inner % 2):
            raise ValueError(
                "evaluation.z_inner_paths must be one or a positive even integer"
            )
        if pair_batch <= 0:
            raise ValueError("evaluation.z_outer_pair_batch must be positive")
        if float(self.evaluation.get("z_ridge_relative", 1e-12)) < 0.0:
            raise ValueError("evaluation.z_ridge_relative must be nonnegative")
        if int(self.model.get("n_steps", 1)) <= 0:
            raise ValueError("model.n_steps must be positive")
        for key in ("states", "continuations", "continuation_batch"):
            if int(self.evaluation.get(key, 1)) <= 0:
                raise ValueError(f"evaluation.{key} must be positive")
        antithetic = bool(self.evaluation.get("antithetic", False))
        if antithetic and int(self.evaluation.get("continuations", 1)) % 2:
            raise ValueError("Antithetic evaluation requires an even continuation count")
        if antithetic and int(self.evaluation.get("continuation_batch", 1)) % 2:
            raise ValueError("Antithetic evaluation requires an even continuation batch")
        holdout = int(self.evaluation.get("holdout_continuations", 0))
        if holdout < 0:
            raise ValueError("evaluation.holdout_continuations must be nonnegative")
        if antithetic and holdout % 2:
            raise ValueError("Antithetic holdout evaluation requires an even continuation count")
        holdout_batch = int(
            self.evaluation.get(
                "holdout_continuation_batch",
                self.evaluation.get("continuation_batch", 1),
            )
        )
        if holdout and holdout_batch <= 0:
            raise ValueError("evaluation.holdout_continuation_batch must be positive")
        if antithetic and holdout and holdout_batch % 2:
            raise ValueError("Antithetic holdout evaluation requires an even continuation batch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "device": self.device,
            "dtype": self.dtype,
            "output_root": self.output_root,
            "model": copy.deepcopy(self.model),
            "policy": copy.deepcopy(self.policy),
            "training": copy.deepcopy(self.training),
            "evaluation": copy.deepcopy(self.evaluation),
            "recovery": copy.deepcopy(self.recovery),
        }

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def output_dir(self, override: str | None = None) -> Path:
        root = Path(override or self.output_root)
        if not root.is_absolute() and self.source_path:
            root = Path(self.source_path).resolve().parent.parent / root
        return root / self.name


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Configuration root must be a mapping")
    return ExperimentConfig.from_mapping(raw, source_path=str(source))
