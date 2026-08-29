import torch

from mf_revision.random import BrownianSpec, DeterministicBrownianBank


def test_state_indexed_bank_is_order_and_batch_invariant() -> None:
    spec = BrownianSpec(
        seed=17, continuations=8, steps=3, brownian_dim=2, antithetic=True
    )
    bank = DeterministicBrownianBank(spec)
    direct = bank.for_states([2, 9])
    reversed_bank = bank.for_states(torch.tensor([9, 2]))
    assert torch.equal(direct[0], reversed_bank[1])
    assert torch.equal(direct[1], reversed_bank[0])
    pairs = direct.reshape(2, 4, 2, 3, 2)
    assert torch.equal(pairs.sum(dim=2), torch.zeros_like(pairs.sum(dim=2)))


def test_first_step_common_future_pairing_flips_only_first_increment() -> None:
    spec = BrownianSpec(
        seed=29,
        continuations=8,
        steps=4,
        brownian_dim=3,
        antithetic=True,
        pairing="first_step_common_future",
    )
    paths = DeterministicBrownianBank(spec).for_state(5).reshape(4, 2, 4, 3)
    assert torch.equal(paths[:, 0, 0, :], -paths[:, 1, 0, :])
    assert torch.equal(paths[:, 0, 1:, :], paths[:, 1, 1:, :])
