from pathlib import Path

from mf_revision.config import ExperimentConfig
from mf_revision.experiments import ExperimentRunner


def test_analytic_pipeline_writes_canonical_outputs(tmp_path: Path) -> None:
    config = ExperimentConfig.from_mapping(
        {
            "name": "test_pipeline",
            "seed": 3,
            "device": "cpu",
            "dtype": "float64",
            "model": {
                "name": "merton",
                "d": 1,
                "T": 0.25,
                "n_steps": 4,
                "r": 0.03,
                "gamma": 2.0,
                "constraint": "unconstrained",
                "alpha": [0.05],
                "covariance": [[0.04]],
            },
            "policy": {"kind": "analytic"},
            "evaluation": {
                "states": 2,
                "continuations": 32,
                "continuation_batch": 8,
                "holdout_continuations": 16,
                "holdout_continuation_batch": 8,
                "antithetic": True,
                "graph_mode": "ol",
            },
            "recovery": {"barrier": False, "curvature_floor": 1e-12},
        }
    )
    runner = ExperimentRunner(config, output_override=str(tmp_path))
    estimate, recovery = runner.pipeline()
    output = tmp_path / "test_pipeline"
    for filename in (
        "resolved_config.yaml",
        "run_card.json",
        "evaluation_states.pt",
        "adjoints.pt",
        "adjoints_holdout.pt",
        "recovery.pt",
        "adjoint_summary.json",
        "adjoint_holdout_summary.json",
        "recovery_summary.json",
        "pointwise.csv",
    ):
        assert (output / filename).exists()
    assert estimate.zeta.shape[-1] == 1
    assert recovery.full_control.shape == recovery.zero_shift_control.shape
    assert "holdout_full_gain_vs_reference" in recovery.diagnostics
