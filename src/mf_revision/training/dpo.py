"""Feedback direct-policy optimization with optional nested checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import torch
import torch.nn as nn

from mf_revision.models.base import PortfolioModel
from mf_revision.policies.factory import save_policy
from mf_revision.simulation import rollout_payoff


@dataclass(slots=True)
class TrainingResult:
    policy: nn.Module
    loss_history: list[float]
    checkpoint_path: Path
    checkpoint_paths: dict[int, Path]


def train_dpo(
    model: PortfolioModel,
    policy: nn.Module,
    config: dict[str, Any],
    *,
    output_dir: str | Path,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    metadata: dict[str, Any] | None = None,
) -> TrainingResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    epochs = int(config.get("epochs", 500))
    batch_size = int(config.get("batch_size", 1024))
    learning_rate = float(config.get("learning_rate", config.get("lr", 1e-4)))
    grad_clip = float(config.get("grad_clip", 1.0))
    print_every = int(config.get("print_every", 25))
    requested = {int(value) for value in config.get("checkpoint_epochs", [])}
    requested.add(epochs)
    if min(requested) <= 0 or max(requested) > epochs:
        raise ValueError("training.checkpoint_epochs must lie in [1, epochs]")

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    history: list[float] = []
    checkpoint_paths: dict[int, Path] = {}
    policy.train()

    generator_device = device.type if device.type != "cpu" else "cpu"
    for epoch in range(1, epochs + 1):
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(int(seed) + epoch)
        state, tau = model.sample_initial_states(
            batch_size, generator=generator, device=device, dtype=dtype
        )
        normals = torch.randn(
            batch_size,
            model.n_steps,
            model.dims.brownian,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        optimizer.zero_grad(set_to_none=True)
        result = rollout_payoff(model, policy, state, tau, normals, graph_mode="cl")
        loss = -result.payoff.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()
        history.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            print(f"[train {epoch:05d}/{epochs}] loss={history[-1]:.8g}")
        if epoch in requested:
            target = output / ("policy.pt" if epoch == epochs else f"checkpoint_e{epoch:04d}.pt")
            save_policy(
                target,
                policy,
                metadata={
                    **dict(metadata or {}),
                    "seed": int(seed),
                    "epoch": epoch,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                },
            )
            checkpoint_paths[epoch] = target

    checkpoint = checkpoint_paths[epochs]
    with (output / "loss_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "loss"])
        writer.writerows((index + 1, value) for index, value in enumerate(history))
    return TrainingResult(
        policy=policy,
        loss_history=history,
        checkpoint_path=checkpoint,
        checkpoint_paths=checkpoint_paths,
    )
