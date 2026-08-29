"""Fixed-latent OL-BPTT harvesting of lambda, full P, Z, and zeta."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn

from mf_revision.adjoints.projection import (
    estimate_brownian_projection,
    paired_projection_units,
)
from mf_revision.models.base import PortfolioModel
from mf_revision.random import BrownianSpec, DeterministicBrownianBank, stable_seed
from mf_revision.simulation import first_reference_step, rollout_payoff
from mf_revision.types import AdjointEstimate


@dataclass(slots=True)
class _Moments:
    total: torch.Tensor
    total_square: torch.Tensor
    count: int = 0

    @classmethod
    def zeros(
        cls, shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype
    ) -> "_Moments":
        return cls(
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
            0,
        )

    def add(self, samples: torch.Tensor) -> None:
        self.total += samples.sum(dim=0)
        self.total_square += samples.square().sum(dim=0)
        self.count += int(samples.shape[0])

    def mean(self) -> torch.Tensor:
        if self.count <= 0:
            raise RuntimeError("No Monte Carlo samples accumulated")
        return self.total / float(self.count)

    def standard_error(self) -> torch.Tensor:
        if self.count <= 1:
            return torch.full_like(self.total, float("nan"))
        mean = self.mean()
        sum_squares = (
            self.total_square - float(self.count) * mean.square()
        ).clamp_min(0.0)
        return torch.sqrt(sum_squares / float(self.count - 1) / float(self.count))


def _gradient(
    payoff: torch.Tensor, state: torch.Tensor, *, create_graph: bool
) -> torch.Tensor:
    return torch.autograd.grad(
        payoff.sum(),
        state,
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=False,
    )[0]


def _gradient_hessian(
    payoff: torch.Tensor, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    gradient = _gradient(payoff, state, create_graph=True)
    rows: list[torch.Tensor] = []
    for row_index in range(state.shape[1]):
        rows.append(
            torch.autograd.grad(
                gradient[:, row_index].sum(),
                state,
                retain_graph=row_index < state.shape[1] - 1,
                create_graph=False,
                allow_unused=False,
            )[0]
        )
    hessian = torch.stack(rows, dim=1)
    return gradient, 0.5 * (hessian + hessian.transpose(1, 2))


def _independent_units(samples: torch.Tensor, *, paired: bool) -> torch.Tensor:
    if not paired:
        return samples
    if samples.shape[0] % 2:
        raise ValueError("Paired batches must contain complete adjacent pairs")
    return samples.reshape(samples.shape[0] // 2, 2, *samples.shape[1:]).mean(dim=1)


def _harvest_lambda_p_one_state(
    model: PortfolioModel,
    policy: nn.Module,
    state: torch.Tensor,
    tau: torch.Tensor,
    *,
    state_id: int,
    bank: DeterministicBrownianBank,
    continuation_batch: int,
    graph_mode: str,
) -> dict[str, torch.Tensor | float | int]:
    spec = bank.spec
    device, dtype = state.device, state.dtype
    n_state, q_brownian = model.dims.state, model.dims.brownian
    if spec.antithetic and continuation_batch % 2:
        raise ValueError("continuation_batch must be even in paired mode")

    with torch.no_grad():
        reference_control = policy(state, tau)
        sigma_ref = model.state_diffusion(state, reference_control)
    moments = {
        "lambda": _Moments.zeros((n_state,), device=device, dtype=dtype),
        "p": _Moments.zeros((n_state, n_state), device=device, dtype=dtype),
        "p_sigma": _Moments.zeros(
            (n_state, q_brownian), device=device, dtype=dtype
        ),
    }
    normals_full = bank.for_state(state_id)
    for start in range(0, spec.continuations, continuation_batch):
        stop = min(spec.continuations, start + continuation_batch)
        normals = normals_full[start:stop].to(device=device, dtype=dtype)
        paths = normals.shape[0]
        state0 = state.expand(paths, -1).clone().requires_grad_(True)
        tau0 = tau.expand(paths, -1).clone()
        with torch.enable_grad():
            payoff0 = rollout_payoff(
                model, policy, state0, tau0, normals, graph_mode=graph_mode
            ).payoff
            lambda0, p0 = _gradient_hessian(payoff0, state0)

        lambda_units = _independent_units(
            lambda0.detach(), paired=spec.antithetic
        )
        p_units = _independent_units(p0.detach(), paired=spec.antithetic)
        p_sigma_units = torch.bmm(
            p_units, sigma_ref.expand(p_units.shape[0], -1, -1)
        )
        moments["lambda"].add(lambda_units)
        moments["p"].add(p_units)
        moments["p_sigma"].add(p_sigma_units)

    return {
        "lambda": moments["lambda"].mean(),
        "p": moments["p"].mean(),
        "p_sigma": moments["p_sigma"].mean(),
        "sigma_ref": sigma_ref[0],
        "reference_control": reference_control[0],
        "lambda_se": moments["lambda"].standard_error(),
        "p_se": moments["p"].standard_error(),
        "p_sigma_se": moments["p_sigma"].standard_error(),
        "independent_units": moments["lambda"].count,
    }


def _inner_future_bank(
    *,
    pair_count: int,
    inner_paths: int,
    future_steps: int,
    brownian_dim: int,
    seed: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if inner_paths <= 0:
        raise ValueError("projection_inner_paths must be positive")
    if future_steps == 0:
        return torch.empty(
            pair_count,
            inner_paths,
            0,
            brownian_dim,
            dtype=dtype,
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if inner_paths == 1:
        return torch.randn(
            pair_count,
            1,
            future_steps,
            brownian_dim,
            generator=generator,
            dtype=dtype,
        )
    if inner_paths % 2:
        raise ValueError("projection_inner_paths must be even when greater than one")
    base = torch.randn(
        pair_count,
        inner_paths // 2,
        future_steps,
        brownian_dim,
        generator=generator,
        dtype=dtype,
    )
    return torch.stack((base, -base), dim=2).reshape(
        pair_count, inner_paths, future_steps, brownian_dim
    )


def _projection_units_one_state(
    model: PortfolioModel,
    policy: nn.Module,
    state: torch.Tensor,
    tau: torch.Tensor,
    *,
    state_id: int,
    bank: DeterministicBrownianBank,
    continuation_batch: int,
    inner_paths: int,
    inner_seed: int,
    graph_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Build independent Brownian-regression units.

    The projection bank contains antithetic outer first-step increments.  For
    ``inner_paths=1`` its common future path is used directly.  For larger
    ``inner_paths``, each outer pair receives an antithetic inner continuation
    bank that is shared between the + and - next states.  Averaging the inner
    continuation gradients approximates the adapted costate at ``t+dt`` before
    the Brownian regression.
    """
    spec = bank.spec
    device, dtype = state.device, state.dtype
    if not spec.antithetic or spec.pairing != "first_step_common_future":
        raise ValueError(
            "The projection bank must use antithetic first_step_common_future pairing"
        )
    if continuation_batch % 2:
        raise ValueError("projection_batch must be even")
    pair_count = spec.continuations // 2
    pair_batch = max(1, continuation_batch // 2)
    future_steps = max(model.n_steps - 1, 0)
    normals_full = bank.for_state(state_id)
    outer_pairs = normals_full.reshape(pair_count, 2, model.n_steps, model.dims.brownian)
    inner = _inner_future_bank(
        pair_count=pair_count,
        inner_paths=inner_paths,
        future_steps=future_steps,
        brownian_dim=model.dims.brownian,
        seed=stable_seed(inner_seed, "projection_inner", int(state_id)),
        dtype=dtype,
    )

    response_parts: list[torch.Tensor] = []
    brownian_parts: list[torch.Tensor] = []
    dt_value: float | None = None
    for start in range(0, pair_count, pair_batch):
        stop = min(pair_count, start + pair_batch)
        pairs = stop - start
        outer = outer_pairs[start:stop].to(device=device, dtype=dtype)
        first_normals = outer[:, :, 0, :].reshape(pairs * 2, model.dims.brownian)
        state0 = state.expand(pairs * 2, -1)
        tau0 = tau.expand(pairs * 2, -1)
        with torch.no_grad():
            next_state, _, dt = first_reference_step(
                model, policy, state0, tau0, first_normals
            )
        if dt_value is None:
            dt_value = float(dt[0, 0].detach().cpu())

        future = inner[start:stop].to(device=device, dtype=dtype)
        future = (
            future[:, None, :, :, :]
            .expand(pairs, 2, inner_paths, future_steps, model.dims.brownian)
            .reshape(pairs * 2 * inner_paths, future_steps, model.dims.brownian)
        )
        state1 = (
            next_state.reshape(pairs, 2, model.dims.state)[:, :, None, :]
            .expand(pairs, 2, inner_paths, model.dims.state)
            .reshape(pairs * 2 * inner_paths, model.dims.state)
            .detach()
            .clone()
            .requires_grad_(True)
        )
        tau1_base = (tau0 - dt).clamp_min(0.0).reshape(pairs, 2, 1)
        tau1 = (
            tau1_base[:, :, None, :]
            .expand(pairs, 2, inner_paths, 1)
            .reshape(pairs * 2 * inner_paths, 1)
        )
        with torch.enable_grad():
            payoff1 = rollout_payoff(
                model,
                policy,
                state1,
                tau1,
                future,
                graph_mode=graph_mode,
                discount_offset=dt[0, 0],
            ).payoff
            lambda1 = _gradient(payoff1, state1, create_graph=False)
        lambda1_adapted = lambda1.reshape(
            pairs, 2, inner_paths, model.dims.state
        ).mean(dim=2)
        response_units = 0.5 * (
            lambda1_adapted[:, 0, :] - lambda1_adapted[:, 1, :]
        )
        delta_w_pair = first_normals.reshape(
            pairs, 2, model.dims.brownian
        ) * dt[0, 0].sqrt()
        brownian_units = 0.5 * (
            delta_w_pair[:, 0, :] - delta_w_pair[:, 1, :]
        )
        response_parts.append(response_units.detach())
        brownian_parts.append(brownian_units.detach())

    if dt_value is None:
        raise RuntimeError("No projection samples accumulated")
    return torch.cat(response_parts, dim=0), torch.cat(brownian_parts, dim=0), dt_value


def _harvest_projection_one_state(
    model: PortfolioModel,
    policy: nn.Module,
    state: torch.Tensor,
    tau: torch.Tensor,
    *,
    state_id: int,
    bank: DeterministicBrownianBank,
    continuation_batch: int,
    inner_paths: int,
    inner_seed: int,
    graph_mode: str,
    p_sigma: torch.Tensor,
    p_sigma_se: torch.Tensor,
    method: str,
    ridge: float,
) -> dict[str, Any]:
    response, brownian, dt_value = _projection_units_one_state(
        model,
        policy,
        state,
        tau,
        state_id=state_id,
        bank=bank,
        continuation_batch=continuation_batch,
        inner_paths=inner_paths,
        inner_seed=inner_seed,
        graph_mode=graph_mode,
    )
    selected_method = str(method).lower().replace("-", "_")
    if selected_method == "nested_crn_ols":
        selected_method = "ols_control_variate"

    variants: dict[str, torch.Tensor] = {}
    se_variants: dict[str, torch.Tensor] = {}
    results = {}
    for estimator in (
        "raw_moment",
        "centered_moment",
        "control_variate",
        "ols_control_variate",
    ):
        try:
            current = estimate_brownian_projection(
                response,
                brownian,
                dt=dt_value,
                p_sigma=p_sigma,
                p_sigma_se=p_sigma_se,
                method=estimator,
                ridge=ridge,
            )
        except ValueError:
            if estimator == selected_method:
                raise
            continue
        results[estimator] = current
        variants[f"{estimator}_z"] = current.z
        variants[f"{estimator}_zeta"] = current.zeta
        se_variants[f"{estimator}_z"] = current.z_se
        se_variants[f"{estimator}_zeta"] = current.zeta_se
    if selected_method not in results:
        raise ValueError(f"Selected projection estimator is unavailable: {selected_method}")
    selected = results[selected_method]
    return {
        "z": selected.z,
        "zeta": selected.zeta,
        "z_se": selected.z_se,
        "zeta_se": selected.zeta_se,
        "projection_units": selected.units,
        "projection_variants": variants,
        "projection_se_variants": se_variants,
        **selected.diagnostics,
    }


def _harvest_one_state(
    model: PortfolioModel,
    policy: nn.Module,
    state: torch.Tensor,
    tau: torch.Tensor,
    *,
    state_id: int,
    bank: DeterministicBrownianBank,
    projection_bank: DeterministicBrownianBank,
    continuation_batch: int,
    projection_batch: int,
    projection_inner_paths: int,
    projection_inner_seed: int,
    graph_mode: str,
    projection_method: str,
    projection_ridge: float,
) -> dict[str, Any]:
    adapted = _harvest_lambda_p_one_state(
        model,
        policy,
        state,
        tau,
        state_id=state_id,
        bank=bank,
        continuation_batch=continuation_batch,
        graph_mode=graph_mode,
    )
    projection = _harvest_projection_one_state(
        model,
        policy,
        state,
        tau,
        state_id=state_id,
        bank=projection_bank,
        continuation_batch=projection_batch,
        inner_paths=projection_inner_paths,
        inner_seed=projection_inner_seed,
        graph_mode=graph_mode,
        p_sigma=adapted["p_sigma"],
        p_sigma_se=adapted["p_sigma_se"],
        method=projection_method,
        ridge=projection_ridge,
    )
    return {**adapted, **projection}


def harvest_adjoint_estimate(
    model: PortfolioModel,
    policy: nn.Module,
    states: torch.Tensor,
    tau: torch.Tensor,
    bank: DeterministicBrownianBank,
    *,
    projection_bank: DeterministicBrownianBank | None = None,
    state_ids: torch.Tensor | None = None,
    continuation_batch: int,
    projection_batch: int | None = None,
    projection_inner_paths: int = 1,
    projection_inner_seed: int | None = None,
    graph_mode: str = "ol",
    projection_method: str = "ols_control_variate",
    projection_ridge: float = 1.0e-12,
    metadata: dict[str, Any] | None = None,
) -> AdjointEstimate:
    """Estimate the adapted tuple with separate P and Brownian-projection banks.

    The recommended full-shift estimator uses an antithetic outer projection
    bank with opposite first Brownian innovations, common future shocks, inner
    conditional averaging, and an OLS control variate anchored at
    ``P sigma_ref``.  ``projection_inner_paths=1`` retains the cheaper common-
    future central-difference estimator; larger values approximate the adapted
    next-step costate before estimating ``Z``.
    """
    if states.ndim != 2 or tau.shape != (states.shape[0], 1):
        raise ValueError("Expected states[B,n] and tau[B,1]")
    if graph_mode not in {"ol", "cl"}:
        raise ValueError("graph_mode must be ol or cl")
    if projection_inner_paths <= 0:
        raise ValueError("projection_inner_paths must be positive")
    if projection_inner_paths > 1 and projection_inner_paths % 2:
        raise ValueError("projection_inner_paths must be even when greater than one")

    if projection_bank is None:
        projection_bank = DeterministicBrownianBank(
            BrownianSpec(
                seed=bank.spec.seed + 104729,
                continuations=bank.spec.continuations,
                steps=bank.spec.steps,
                brownian_dim=bank.spec.brownian_dim,
                antithetic=True,
                dtype=bank.spec.dtype,
                pairing="first_step_common_future",
            )
        )
    projection_batch = int(projection_batch or continuation_batch)
    projection_inner_seed = int(
        projection_inner_seed
        if projection_inner_seed is not None
        else projection_bank.spec.seed + 130363
    )
    for current in (bank.spec, projection_bank.spec):
        if current.steps != model.n_steps or current.brownian_dim != model.dims.brownian:
            raise ValueError("Brownian specification does not match model dimensions")
    if state_ids is None:
        state_ids = torch.arange(states.shape[0], device=states.device, dtype=torch.long)
    if state_ids.numel() != states.shape[0]:
        raise ValueError("state_ids length mismatch")

    parameters = list(policy.parameters())
    flags = [parameter.requires_grad for parameter in parameters]
    was_training = policy.training
    for parameter in parameters:
        parameter.requires_grad_(False)
    policy.eval()
    try:
        rows = [
            _harvest_one_state(
                model,
                policy,
                states[index : index + 1],
                tau[index : index + 1],
                state_id=int(state_ids[index]),
                bank=bank,
                projection_bank=projection_bank,
                continuation_batch=int(continuation_batch),
                projection_batch=projection_batch,
                projection_inner_paths=int(projection_inner_paths),
                projection_inner_seed=projection_inner_seed,
                graph_mode=graph_mode,
                projection_method=projection_method,
                projection_ridge=float(projection_ridge),
            )
            for index in range(states.shape[0])
        ]
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)
        policy.train(was_training)

    def stack(key: str) -> torch.Tensor:
        return torch.stack([row[key] for row in rows])

    variant_keys = sorted(
        set.intersection(
            *[set(row["projection_variants"].keys()) for row in rows]
        )
    )
    projection_variants = {
        key: torch.stack([row["projection_variants"][key] for row in rows])
        for key in variant_keys
    }
    se_variant_keys = sorted(
        set.intersection(
            *[set(row["projection_se_variants"].keys()) for row in rows]
        )
    )
    projection_se_variants = {
        key: torch.stack([row["projection_se_variants"][key] for row in rows])
        for key in se_variant_keys
    }

    identity_max = max(float(row["zeta_identity_error"]) for row in rows)
    gram_conditions = [float(row["projection_gram_condition"]) for row in rows]
    finite_conditions = [value for value in gram_conditions if torch.isfinite(torch.tensor(value))]
    meta = dict(metadata or {})
    meta.update(
        {
            "graph_mode": graph_mode,
            "continuations": bank.spec.continuations,
            "independent_mc_units": int(rows[0]["independent_units"]),
            "steps": bank.spec.steps,
            "brownian_dim": bank.spec.brownian_dim,
            "antithetic": bank.spec.antithetic,
            "brownian_pairing": bank.spec.pairing,
            "continuation_batch": int(continuation_batch),
            "projection_continuations": projection_bank.spec.continuations,
            "projection_independent_mc_units": int(rows[0]["projection_units"]),
            "projection_seed": projection_bank.spec.seed,
            "projection_antithetic": projection_bank.spec.antithetic,
            "projection_pairing": projection_bank.spec.pairing,
            "projection_batch": projection_batch,
            "projection_inner_paths": int(projection_inner_paths),
            "projection_total_future_rollouts": int(
                projection_bank.spec.continuations * projection_inner_paths
            ),
            "projection_inner_seed": projection_inner_seed,
            "z_estimator": projection_method,
            "zeta_estimator": "direct_residual_" + projection_method,
            "projection_ridge": float(projection_ridge),
            "projection_gram_condition_max": (
                max(finite_conditions) if finite_conditions else float("nan")
            ),
            "projection_gram_condition_mean": (
                sum(finite_conditions) / len(finite_conditions)
                if finite_conditions
                else float("nan")
            ),
            "zeta_identity_max_abs": identity_max,
        }
    )
    return AdjointEstimate(
        states=states.detach(),
        tau=tau.detach(),
        lambda_=stack("lambda"),
        p=stack("p"),
        z=stack("z"),
        zeta=stack("zeta"),
        sigma_ref=stack("sigma_ref"),
        reference_control=stack("reference_control"),
        lambda_se=stack("lambda_se"),
        p_se=stack("p_se"),
        z_se=stack("z_se"),
        zeta_se=stack("zeta_se"),
        projection_variants=projection_variants,
        projection_se_variants=projection_se_variants,
        metadata=meta,
    )
