import torch

from mf_revision.adjoints import harvest_adjoint_estimate
from mf_revision.models import AffineFactorModel, MertonModel
from mf_revision.random import BrownianSpec, DeterministicBrownianBank


def _merton_model(steps: int = 16) -> MertonModel:
    return MertonModel(
        {
            "name": "merton",
            "d": 1,
            "T": 0.5,
            "n_steps": steps,
            "r": 0.03,
            "rho": 0.0,
            "gamma": 2.0,
            "constraint": "unconstrained",
            "alpha": [0.06],
            "covariance": [[0.04]],
        },
        device=torch.device("cpu"),
        dtype=torch.float64,
    )


def test_harvest_is_continuation_batch_invariant() -> None:
    model = _merton_model(steps=8)
    policy = model.analytical_policy()
    states = torch.tensor([[1.0]], dtype=torch.float64)
    tau = torch.tensor([[0.5]], dtype=torch.float64)
    bank = DeterministicBrownianBank(
        BrownianSpec(123, 128, 8, 1, True, torch.float64)
    )
    left = harvest_adjoint_estimate(
        model, policy, states, tau, bank, continuation_batch=16
    )
    right = harvest_adjoint_estimate(
        model, policy, states, tau, bank, continuation_batch=64
    )
    for name in ("lambda_", "p", "z", "zeta"):
        assert torch.allclose(getattr(left, name), getattr(right, name), rtol=1e-13, atol=1e-13)


def test_exact_merton_full_tuple_is_uncertainty_consistent() -> None:
    model = _merton_model(steps=16)
    policy = model.analytical_policy()
    states = torch.tensor([[0.8], [1.2]], dtype=torch.float64)
    tau = torch.tensor([[0.25], [0.5]], dtype=torch.float64)
    estimate = harvest_adjoint_estimate(
        model,
        policy,
        states,
        tau,
        DeterministicBrownianBank(
            BrownianSpec(456, 1024, 16, 1, True, torch.float64)
        ),
        continuation_batch=128,
    )
    exact = model.exact_adjoint_fields(states, tau)
    lambda_error = torch.sqrt(torch.mean((estimate.lambda_x - exact["lambda"]).square()))
    lambda_scale = torch.sqrt(torch.mean(exact["lambda"].square()))
    p_error = torch.sqrt(torch.mean((estimate.p_xrow - exact["P_wealth_row"]).square()))
    p_scale = torch.sqrt(torch.mean(exact["P_wealth_row"].square()))
    assert float(lambda_error / lambda_scale) < 0.02
    assert float(p_error / p_scale) < 0.02
    assert estimate.zeta_se is not None
    # Exact zeta is zero. A finite-MC estimate should be statistically compatible with zero.
    assert torch.all(
        estimate.zeta_x.abs() <= 4.0 * estimate.zeta_se[:, 0, :].abs() + 0.02
    )
    assert estimate.metadata["zeta_identity_max_abs"] < 1e-12


def test_exact_affine_harvests_mixed_wealth_row() -> None:
    model = AffineFactorModel(
        {
            "name": "affine_factor",
            "d": 1,
            "k": 1,
            "exact_one_factor": True,
            "constraint": "unconstrained",
            "T": 1.0,
            "n_steps": 16,
            "r": 0.03,
            "gamma": 2.0,
            "a0": 0.0,
            "a1": 0.2,
            "sigma": 0.2,
            "kappa": 2.0,
            "theta": 0.2,
            "nu": 0.3,
            "corr": 0.1,
        },
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    states = torch.tensor([[1.0, 0.2]], dtype=torch.float64)
    tau = torch.tensor([[0.5]], dtype=torch.float64)
    estimate = harvest_adjoint_estimate(
        model,
        model.analytical_policy(),
        states,
        tau,
        DeterministicBrownianBank(
            BrownianSpec(789, 1024, 16, 2, True, torch.float64)
        ),
        continuation_batch=128,
    )
    exact = model.exact_adjoint_fields(states, tau)
    # VXY is an analytical diagnostic for the harvested mixed PMP adjoint; the
    # code deliberately does not declare it to be the exact PXY target.
    relative = (estimate.p_xrow[:, 1:] - exact["VXY"]).abs() / exact[
        "VXY"
    ].abs().clamp_min(1e-12)
    assert float(relative.max()) < 0.15
    assert estimate.metadata["zeta_identity_max_abs"] < 1e-12


def test_nested_crn_ols_recovers_exact_merton_z() -> None:
    model = _merton_model(steps=16)
    policy = model.analytical_policy()
    states = torch.tensor([[0.9], [1.1]], dtype=torch.float64)
    tau = torch.tensor([[0.3], [0.5]], dtype=torch.float64)
    main_bank = DeterministicBrownianBank(
        BrownianSpec(900, 512, 16, 1, True, torch.float64)
    )
    projection_bank = DeterministicBrownianBank(
        BrownianSpec(
            901,
            128,
            16,
            1,
            True,
            torch.float64,
            pairing="first_step_common_future",
        )
    )
    estimate = harvest_adjoint_estimate(
        model,
        policy,
        states,
        tau,
        main_bank,
        projection_bank=projection_bank,
        continuation_batch=64,
        projection_batch=32,
        projection_inner_paths=8,
        projection_inner_seed=902,
        projection_method="ols_control_variate",
    )
    exact = model.exact_adjoint_fields(states, tau)
    relative = torch.sqrt(torch.mean((estimate.z_x - exact["Z_wealth"]).square())) / torch.sqrt(
        torch.mean(exact["Z_wealth"].square())
    )
    assert float(relative) < 0.02
    assert float(torch.sqrt(torch.mean(estimate.zeta_x.square()))) < 0.01
    assert estimate.metadata["projection_inner_paths"] == 8
    assert estimate.metadata["zeta_identity_max_abs"] < 1e-12


def test_nested_projection_is_pair_batch_invariant() -> None:
    model = _merton_model(steps=8)
    policy = model.analytical_policy()
    states = torch.tensor([[1.0]], dtype=torch.float64)
    tau = torch.tensor([[0.5]], dtype=torch.float64)
    main_bank = DeterministicBrownianBank(
        BrownianSpec(910, 128, 8, 1, True, torch.float64)
    )
    projection_bank = DeterministicBrownianBank(
        BrownianSpec(
            911,
            128,
            8,
            1,
            True,
            torch.float64,
            pairing="first_step_common_future",
        )
    )
    left = harvest_adjoint_estimate(
        model,
        policy,
        states,
        tau,
        main_bank,
        projection_bank=projection_bank,
        continuation_batch=32,
        projection_batch=16,
        projection_inner_paths=8,
        projection_inner_seed=912,
    )
    right = harvest_adjoint_estimate(
        model,
        policy,
        states,
        tau,
        main_bank,
        projection_bank=projection_bank,
        continuation_batch=64,
        projection_batch=64,
        projection_inner_paths=8,
        projection_inner_seed=912,
    )
    for name in ("lambda_", "p", "z", "zeta"):
        assert torch.allclose(
            getattr(left, name), getattr(right, name), rtol=1e-12, atol=1e-12
        )
