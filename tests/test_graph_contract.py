import torch
import torch.nn as nn

from mf_revision.simulation import policy_control


class QuadraticFeedback(nn.Module):
    def latent(self, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        del tau
        return 2.0 * state[:, 0:1]

    def decode(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return state[:, 0:1] * latent

    def forward(self, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return self.decode(state, self.latent(state, tau))


def _first_second(mode: str) -> tuple[float, float]:
    state = torch.tensor([[3.0]], dtype=torch.float64, requires_grad=True)
    tau = torch.ones(1, 1, dtype=torch.float64)
    value = policy_control(QuadraticFeedback(), state, tau, graph_mode=mode)
    first = torch.autograd.grad(value.sum(), state, create_graph=True)[0]
    second = (
        torch.autograd.grad(first.sum(), state)[0]
        if first.requires_grad
        else torch.zeros_like(state)
    )
    return float(first.detach()), float(second.detach())


def test_ol_removes_actor_derivative_but_retains_chart_derivative() -> None:
    first_ol, second_ol = _first_second("ol")
    first_cl, second_cl = _first_second("cl")
    # OL: latent=6 is fixed and decode=x*latent.
    assert first_ol == 6.0
    assert second_ol == 0.0
    # CL: decode=x*(2x)=2x^2.
    assert first_cl == 12.0
    assert second_cl == 4.0
