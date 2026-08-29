from __future__ import annotations

import torch

from mf_revision.models.factory import build_model
from mf_revision.recovery.consumption import solve_consumption_barrier


def test_consumption_barrier_stays_strict_and_converges() -> None:
    model = build_model(
        {
            "name": "merton",
            "d": 1,
            "T": 1.0,
            "n_steps": 4,
            "r": 0.03,
            "rho": 0.1,
            "gamma": 2.0,
            "constraint": "simplex",
            "leverage_cap": 1.0,
            "consumption": True,
            "consumption_rate_min": 1e-8,
            "consumption_rate_max": 0.7,
            "alpha": [0.05],
            "covariance": [[0.04]],
        },
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    state = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    lambda_x = torch.tensor([[4.0], [0.8]], dtype=torch.float64)
    result = solve_consumption_barrier(
        model, state, lambda_x, epsilon=1e-6, tolerance=1e-9, max_iterations=100
    )
    lower, upper = model.consumption_bounds(state)
    assert bool((result.consumption > lower).all())
    assert bool((result.consumption < upper).all())
    assert float(result.residual.max()) < 1e-7
    assert bool(result.converged.all())
