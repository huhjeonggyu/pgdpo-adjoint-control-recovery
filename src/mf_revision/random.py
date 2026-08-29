"""State-indexed deterministic Brownian continuation banks."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import torch


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**63 - 1)


_PAIRINGS = {"full_path", "first_step_common_future"}


@dataclass(frozen=True, slots=True)
class BrownianSpec:
    seed: int
    continuations: int
    steps: int
    brownian_dim: int
    antithetic: bool = False
    dtype: torch.dtype = torch.float64
    pairing: str = "full_path"


class DeterministicBrownianBank:
    """Generate standard-normal paths by immutable state ID.

    The generated bank is independent of state batching, continuation batching,
    device, and call order. In paired mode, consecutive paths form one Monte
    Carlo unit.

    ``full_path`` reproduces the usual antithetic construction ``(eps,-eps)``.
    ``first_step_common_future`` flips only the first Brownian innovation and
    holds every later innovation fixed within the pair. The latter is a
    common-random-number coupling designed for the one-step martingale
    projection: future continuation noise cancels in the pair difference while
    each path retains the correct Brownian marginal law.
    """

    def __init__(self, spec: BrownianSpec) -> None:
        if min(spec.continuations, spec.steps, spec.brownian_dim) <= 0:
            raise ValueError("continuations, steps, and brownian_dim must be positive")
        pairing = str(spec.pairing).lower()
        if pairing not in _PAIRINGS:
            raise ValueError(f"pairing must be one of {sorted(_PAIRINGS)}")
        if spec.antithetic and spec.continuations % 2:
            raise ValueError("Paired mode requires an even continuation count")
        if pairing == "first_step_common_future" and not spec.antithetic:
            raise ValueError("first_step_common_future requires antithetic=True")
        self.spec = BrownianSpec(
            seed=spec.seed,
            continuations=spec.continuations,
            steps=spec.steps,
            brownian_dim=spec.brownian_dim,
            antithetic=spec.antithetic,
            dtype=spec.dtype,
            pairing=pairing,
        )

    def for_state(self, state_id: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stable_seed(self.spec.seed, "brownian", int(state_id)))
        tail = (self.spec.steps, self.spec.brownian_dim)
        if not self.spec.antithetic:
            return torch.randn(
                self.spec.continuations,
                *tail,
                generator=generator,
                dtype=self.spec.dtype,
                device="cpu",
            )
        half = self.spec.continuations // 2
        if self.spec.pairing == "full_path":
            base = torch.randn(half, *tail, generator=generator, dtype=self.spec.dtype)
            return torch.stack((base, -base), dim=1).reshape(
                self.spec.continuations, *tail
            )

        first = torch.randn(
            half,
            1,
            self.spec.brownian_dim,
            generator=generator,
            dtype=self.spec.dtype,
        )
        if self.spec.steps == 1:
            plus = first
            minus = -first
        else:
            future = torch.randn(
                half,
                self.spec.steps - 1,
                self.spec.brownian_dim,
                generator=generator,
                dtype=self.spec.dtype,
            )
            plus = torch.cat((first, future), dim=1)
            minus = torch.cat((-first, future), dim=1)
        return torch.stack((plus, minus), dim=1).reshape(
            self.spec.continuations, *tail
        )

    def for_states(self, state_ids: torch.Tensor | list[int]) -> torch.Tensor:
        ids = (
            state_ids.detach().cpu().tolist()
            if isinstance(state_ids, torch.Tensor)
            else state_ids
        )
        return torch.stack([self.for_state(int(index)) for index in ids], dim=0)
