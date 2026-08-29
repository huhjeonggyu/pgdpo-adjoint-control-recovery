"""Canonical train -> harvest -> recover experiment pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import time
import torch
import yaml

from mf_revision.adjoints import harvest_adjoint_estimate
from mf_revision.config import ExperimentConfig
from mf_revision.models import build_model
from mf_revision.policies import build_policy, load_policy_checkpoint
from mf_revision.random import BrownianSpec, DeterministicBrownianBank
from mf_revision.recovery import recover_controls
from mf_revision.runtime import (
    environment_card,
    file_sha256,
    resolve_device,
    resolve_dtype,
    set_global_seed,
    write_json,
)
from mf_revision.training import train_dpo
from mf_revision.types import AdjointEstimate, RecoveryResult
from .reporting import adjoint_summary, write_summaries
from .states import sample_evaluation_states


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, *, output_override: str | None = None) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.dtype = resolve_dtype(config.dtype)
        self.output_dir = config.output_dir(output_override)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        set_global_seed(config.seed)
        self.model = build_model(config.model, device=self.device, dtype=self.dtype)
        self.policy = build_policy(self.model, config.policy, device=self.device, dtype=self.dtype)
        self.project_root = Path(__file__).resolve().parents[3]
        self._write_run_card()

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / "policy.pt"

    @property
    def configured_checkpoint_path(self) -> Path | None:
        value = self.config.policy.get("checkpoint")
        if not value:
            return None
        path = Path(str(value)).expanduser()
        if not path.is_absolute() and self.config.source_path:
            path = Path(self.config.source_path).resolve().parent / path
        return path.resolve()

    @property
    def adjoint_path(self) -> Path:
        return self.output_dir / "adjoints.pt"

    @property
    def recovery_path(self) -> Path:
        return self.output_dir / "recovery.pt"

    @property
    def holdout_adjoint_path(self) -> Path:
        return self.output_dir / "adjoints_holdout.pt"

    def _write_run_card(self) -> None:
        with (self.output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.config.to_dict(), handle, sort_keys=False)
        write_json(
            self.output_dir / "run_card.json",
            {
                "config_digest": self.config.digest(),
                "environment": environment_card(self.project_root),
                "device": str(self.device),
                "dtype": str(self.dtype),
                "dimensions": {
                    "state": self.model.dims.state,
                    "risky": self.model.dims.risky,
                    "brownian": self.model.dims.brownian,
                    "factor": self.model.dims.factor,
                    "control": self.model.dims.control,
                },
                "mathematical_default": "full_shift",
            },
        )

    def train(self) -> Path:
        if str(self.config.policy.get("kind", "mlp")).lower() == "analytic":
            print("[train] analytical policy selected; skipped")
            return self.checkpoint_path
        external = self.configured_checkpoint_path
        if external is not None and external.exists() and not bool(
            self.config.training.get("force", False)
        ):
            print(f"[train] configured checkpoint exists; skipped: {external}")
            return external
        started = time.perf_counter()
        result = train_dpo(
            self.model,
            self.policy,
            self.config.training,
            output_dir=self.output_dir,
            seed=self.config.seed,
            device=self.device,
            dtype=self.dtype,
            metadata={"config_digest": self.config.digest()},
        )
        write_json(
            self.output_dir / "timing_train.json",
            {"training_seconds": time.perf_counter() - started},
        )
        return result.checkpoint_path

    def load_checkpoint(self, checkpoint: str | Path | None = None) -> None:
        if str(self.config.policy.get("kind", "mlp")).lower() == "analytic":
            return
        if checkpoint is not None:
            path = Path(checkpoint).expanduser().resolve()
        elif self.configured_checkpoint_path is not None:
            path = self.configured_checkpoint_path
        else:
            path = self.checkpoint_path
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}. Run train first.")
        metadata = load_policy_checkpoint(
            path,
            self.policy,
            map_location=self.device,
            strict=bool(self.config.policy.get("checkpoint_strict", True)),
        )
        print(f"[policy] loaded {path} ({file_sha256(path)[:16]})")
        if metadata:
            write_json(self.output_dir / "loaded_checkpoint_metadata.json", metadata)

    def evaluation_states(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        evaluation = self.config.evaluation
        mode = str(evaluation.get("state_mode", "iid")).lower().replace("-", "_")
        role = str(evaluation.get("state_role", mode))
        path_value = evaluation.get("states_file")
        path = (
            Path(str(path_value)).expanduser()
            if path_value
            else self.output_dir / f"evaluation_states_{role}.pt"
        )
        if not path.is_absolute() and self.config.source_path:
            path = Path(self.config.source_path).resolve().parent / path
        count = int(evaluation.get("states", 128))
        if path.exists():
            payload = torch.load(path, map_location=self.device, weights_only=False)
            states = payload["states"].to(device=self.device, dtype=self.dtype)
            tau = payload["tau"].to(device=self.device, dtype=self.dtype)
            state_ids = payload["state_ids"].to(device=self.device, dtype=torch.long)
            if states.shape != (count, self.model.dims.state):
                raise ValueError("Saved evaluation states conflict with this configuration")
            return states, tau, state_ids
        generator_device = self.device.type if self.device.type != "cpu" else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(int(evaluation.get("state_seed", self.config.seed + 1000)))
        states, tau = sample_evaluation_states(
            self.model,
            count,
            evaluation,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        state_ids = torch.arange(count, device=self.device, dtype=torch.long)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "states": states.cpu(),
            "tau": tau.cpu(),
            "state_ids": state_ids.cpu(),
            "state_mode": mode,
        }
        torch.save(payload, path)
        # Backward-compatible canonical filename used by the original clean
        # pipeline and by lightweight downstream scripts.  Split/edge jobs keep
        # their role-specific files and do not overwrite the IID alias.
        if mode in {"iid", "train", "default"} and path_value is None:
            alias = self.output_dir / "evaluation_states.pt"
            if alias != path:
                torch.save(payload, alias)
        return states, tau, state_ids

    def _harvest_with_spec(
        self,
        *,
        states: torch.Tensor,
        tau: torch.Tensor,
        state_ids: torch.Tensor,
        spec: BrownianSpec,
        continuation_batch: int,
        checkpoint: str | Path | None,
        role: str,
    ) -> AdjointEstimate:
        checkpoint_path = (
            Path(checkpoint).expanduser().resolve()
            if checkpoint
            else (self.configured_checkpoint_path or self.checkpoint_path)
        )
        policy_kind = str(self.config.policy.get("kind", "mlp")).lower()
        evaluation = self.config.evaluation
        requested = str(evaluation.get("z_estimator", "raw_moment")).lower().replace("-", "_")
        aliases = {
            "raw": "raw_moment",
            "legacy": "raw_moment",
            "centered": "centered_moment",
            "cv": "control_variate",
            "ols": "ols_control_variate",
            "nested": "nested_crn_ols",
            "nested_ols": "nested_crn_ols",
            "crn_ols": "nested_crn_ols",
        }
        requested = aliases.get(requested, requested)
        projection_method = (
            "ols_control_variate" if requested == "nested_crn_ols" else requested
        )
        inner_paths = int(
            evaluation.get("z_inner_paths", 8 if requested == "nested_crn_ols" else 1)
        )
        outer_paths = int(
            evaluation.get("z_outer_paths", evaluation.get("continuations", 8192))
        )
        outer_pair_batch = int(evaluation.get("z_outer_pair_batch", 32))
        if role == "holdout":
            projection_seed = int(
                evaluation.get(
                    "holdout_z_brownian_seed",
                    evaluation.get("z_brownian_seed", spec.seed + 104729) + 1000003,
                )
            )
            inner_seed = int(
                evaluation.get("holdout_z_inner_seed", projection_seed + 130363)
            )
        else:
            projection_seed = int(
                evaluation.get("z_brownian_seed", spec.seed + 104729)
            )
            inner_seed = int(evaluation.get("z_inner_seed", projection_seed + 130363))
        projection_bank = DeterministicBrownianBank(
            BrownianSpec(
                seed=projection_seed,
                continuations=outer_paths,
                steps=self.model.n_steps,
                brownian_dim=self.model.dims.brownian,
                antithetic=True,
                dtype=self.dtype,
                pairing="first_step_common_future",
            )
        )
        return harvest_adjoint_estimate(
            self.model,
            self.policy,
            states,
            tau,
            DeterministicBrownianBank(spec),
            projection_bank=projection_bank,
            state_ids=state_ids,
            continuation_batch=continuation_batch,
            projection_batch=2 * outer_pair_batch,
            projection_inner_paths=inner_paths,
            projection_inner_seed=inner_seed,
            graph_mode=str(evaluation.get("graph_mode", "ol")).lower(),
            projection_method=projection_method,
            projection_ridge=float(evaluation.get("z_ridge_relative", 1.0e-12)),
            metadata={
                "config_digest": self.config.digest(),
                "checkpoint": str(checkpoint_path) if policy_kind == "mlp" else "analytic",
                "checkpoint_sha256": (
                    file_sha256(checkpoint_path) if policy_kind == "mlp" else None
                ),
                "brownian_seed": spec.seed,
                "state_ids": state_ids.cpu().tolist(),
                "sample_role": role,
                "state_mode": str(evaluation.get("state_mode", "iid")),
                "requested_z_estimator": requested,
            },
        )

    def harvest(self, checkpoint: str | Path | None = None) -> AdjointEstimate:
        self.load_checkpoint(checkpoint)
        states, tau, state_ids = self.evaluation_states()
        evaluation = self.config.evaluation
        spec = BrownianSpec(
            seed=int(evaluation.get("brownian_seed", self.config.seed + 2000)),
            continuations=int(evaluation.get("continuations", 8192)),
            steps=self.model.n_steps,
            brownian_dim=self.model.dims.brownian,
            antithetic=bool(evaluation.get("antithetic", False)),
            dtype=self.dtype,
        )
        started = time.perf_counter()
        estimate = self._harvest_with_spec(
            states=states,
            tau=tau,
            state_ids=state_ids,
            spec=spec,
            continuation_batch=int(evaluation.get("continuation_batch", 256)),
            checkpoint=checkpoint,
            role="estimation",
        )
        estimate.save(self.adjoint_path)
        write_summaries(self.output_dir, self.model, estimate)
        write_json(
            self.output_dir / "timing_harvest.json",
            {"harvest_seconds": time.perf_counter() - started},
        )
        print(f"[harvest] saved {self.adjoint_path}")
        return estimate

    def harvest_holdout(
        self, checkpoint: str | Path | None = None
    ) -> AdjointEstimate | None:
        evaluation = self.config.evaluation
        continuations = int(evaluation.get("holdout_continuations", 0))
        if continuations <= 0:
            return None
        self.load_checkpoint(checkpoint)
        states, tau, state_ids = self.evaluation_states()
        main_seed = int(evaluation.get("brownian_seed", self.config.seed + 2000))
        spec = BrownianSpec(
            seed=int(evaluation.get("holdout_brownian_seed", main_seed + 1)),
            continuations=continuations,
            steps=self.model.n_steps,
            brownian_dim=self.model.dims.brownian,
            antithetic=bool(evaluation.get("antithetic", False)),
            dtype=self.dtype,
        )
        started = time.perf_counter()
        estimate = self._harvest_with_spec(
            states=states,
            tau=tau,
            state_ids=state_ids,
            spec=spec,
            continuation_batch=int(
                evaluation.get(
                    "holdout_continuation_batch",
                    evaluation.get("continuation_batch", 256),
                )
            ),
            checkpoint=checkpoint,
            role="holdout",
        )
        estimate.save(self.holdout_adjoint_path)
        write_json(
            self.output_dir / "adjoint_holdout_summary.json",
            adjoint_summary(self.model, estimate),
        )
        write_json(
            self.output_dir / "timing_holdout.json",
            {"holdout_harvest_seconds": time.perf_counter() - started},
        )
        print(f"[harvest holdout] saved {self.holdout_adjoint_path}")
        return estimate

    def recover(
        self,
        adjoint_path: str | Path | None = None,
        *,
        holdout_adjoint_path: str | Path | None = None,
    ) -> RecoveryResult:
        path = Path(adjoint_path) if adjoint_path else self.adjoint_path
        estimate = AdjointEstimate.load(path, map_location=self.device)
        holdout_path = (
            Path(holdout_adjoint_path)
            if holdout_adjoint_path
            else self.holdout_adjoint_path
        )
        holdout = (
            AdjointEstimate.load(holdout_path, map_location=self.device)
            if holdout_path.exists()
            else None
        )
        started = time.perf_counter()
        result = recover_controls(
            self.model, estimate, self.config.recovery, holdout_estimate=holdout
        )
        result.save(self.recovery_path)
        write_summaries(self.output_dir, self.model, estimate, result)
        write_json(
            self.output_dir / "timing_recovery.json",
            {"recovery_seconds": time.perf_counter() - started},
        )
        print(f"[recover] saved {self.recovery_path}")
        return result

    def pipeline(self) -> tuple[AdjointEstimate, RecoveryResult]:
        if str(self.config.policy.get("kind", "mlp")).lower() == "mlp":
            self.train()
        estimate = self.harvest()
        self.harvest_holdout()
        return estimate, self.recover()

    def inspect(self) -> dict[str, Any]:
        card = {
            "name": self.config.name,
            "output_dir": str(self.output_dir),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "model": self.model.name,
            "state_dim": self.model.dims.state,
            "risky_dim": self.model.dims.risky,
            "brownian_dim": self.model.dims.brownian,
            "constraint": self.model.constraint,
            "full_shift_default": True,
            "n_steps": self.model.n_steps,
        }
        print(card)
        return card
