import torch

from mf_revision.recovery import solve_qp


def test_unconstrained_qp() -> None:
    g = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
    q = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]], dtype=torch.float64)
    solution = solve_qp(g, q, constraint="unconstrained", curvature_floor=0.0)
    assert torch.allclose(solution.control, torch.tensor([[1.0, -0.25]], dtype=torch.float64))
    assert float(solution.projected_gradient_residual.max()) < 1e-12


def test_orthant_qp() -> None:
    g = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
    q = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    solution = solve_qp(g, q, constraint="orthant", curvature_floor=0.0)
    assert torch.allclose(solution.control, torch.tensor([[2.0, 0.0]], dtype=torch.float64))
    assert float(solution.projected_gradient_residual.max()) < 1e-12


def test_simplex_qp() -> None:
    g = torch.tensor([[2.0, 1.0]], dtype=torch.float64)
    q = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    solution = solve_qp(g, q, constraint="simplex", cap=1.0, curvature_floor=0.0)
    assert torch.allclose(solution.control, torch.tensor([[1.0, 0.0]], dtype=torch.float64))
    assert float(solution.projected_gradient_residual.max()) < 1e-10
