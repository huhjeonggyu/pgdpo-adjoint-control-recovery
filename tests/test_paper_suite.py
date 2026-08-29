from __future__ import annotations

from pathlib import Path
import yaml

from mf_revision.experiments.suite import PaperSuite


def test_suite_group_selection_adds_reuse_dependencies(tmp_path: Path) -> None:
    manifest = {
        "output_root": "runs",
        "catalog": "missing_catalog.json",
        "defaults": {
            "device": "cpu",
            "dtype": "float64",
            "model": {
                "name": "merton",
                "d": 1,
                "T": 0.25,
                "n_steps": 2,
                "r": 0.03,
                "gamma": 2.0,
                "constraint": "unconstrained",
                "alpha": [0.05],
                "covariance": [[0.04]],
            },
            "policy": {"kind": "analytic"},
            "evaluation": {
                "states": 1,
                "continuations": 2,
                "continuation_batch": 2,
                "z_outer_paths": 2,
                "z_inner_paths": 1,
            },
        },
        "jobs": [
            {"name": "parent", "groups": ["base"], "stage": "pipeline"},
            {
                "name": "child",
                "groups": ["audit"],
                "stage": "init_audit",
                "reuse_from": "parent",
            },
        ],
    }
    path = tmp_path / "paper" / "suite.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    suite = PaperSuite(path)
    assert [job.name for job in suite.selected(["audit"])] == ["parent", "child"]


def test_suite_loads_job_config_file(tmp_path: Path) -> None:
    paper_dir = tmp_path / "paper"
    config_dir = paper_dir / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "exact.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "device": "cpu",
                "dtype": "float64",
                "model": {
                    "name": "merton",
                    "d": 1,
                    "T": 0.25,
                    "n_steps": 2,
                    "r": 0.03,
                    "gamma": 2.0,
                    "constraint": "unconstrained",
                    "alpha": [0.05],
                    "covariance": [[0.04]],
                },
                "policy": {"kind": "analytic"},
                "evaluation": {
                    "states": 1,
                    "continuations": 2,
                    "continuation_batch": 2,
                    "z_outer_paths": 2,
                    "z_inner_paths": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = paper_dir / "suite.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "output_root": "runs",
                "jobs": [
                    {
                        "name": "exact",
                        "config": "configs/exact.yaml",
                        "groups": ["smoke"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    suite = PaperSuite(manifest_path)
    config = suite.config_for(suite.jobs[0])
    assert config.name == "exact"
    assert config.model["d"] == 1
    assert config.output_root == str((tmp_path / "runs").resolve())


def test_suite_plan_reports_missing_legacy_assets(tmp_path: Path) -> None:
    manifest = {
        "output_root": "runs",
        "catalog": "paper/missing_catalog.json",
        "defaults": {
            "device": "cpu",
            "dtype": "float64",
            "model": {
                "name": "merton",
                "d": 1,
                "T": 0.25,
                "n_steps": 2,
                "r": 0.03,
                "gamma": 2.0,
                "constraint": "unconstrained",
                "alpha": [0.05],
                "covariance": [[0.04]],
            },
            "policy": {"kind": "mlp"},
            "evaluation": {
                "states": 1,
                "continuations": 2,
                "continuation_batch": 2,
                "z_outer_paths": 2,
                "z_inner_paths": 1,
            },
        },
        "jobs": [
            {
                "name": "learned",
                "groups": ["paper"],
                "checkpoint_alias": "missing_checkpoint",
            }
        ],
    }
    path = tmp_path / "paper" / "suite.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    suite = PaperSuite(path)
    assert suite.plan()[0]["missing_assets"] == ["missing_checkpoint"]


def test_suite_reports_catalog_alias_with_missing_target(tmp_path: Path) -> None:
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir(parents=True)
    catalog_path = paper_dir / "catalog.json"
    catalog_path.write_text(
        '{"checkpoint": "/definitely/not/a/real/checkpoint.pt"}',
        encoding="utf-8",
    )
    manifest_path = paper_dir / "suite.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "output_root": "runs",
                "catalog": "paper/catalog.json",
                "defaults": {
                    "device": "cpu",
                    "dtype": "float64",
                    "model": {
                        "name": "merton",
                        "d": 1,
                        "T": 0.25,
                        "n_steps": 2,
                        "r": 0.03,
                        "gamma": 2.0,
                        "constraint": "unconstrained",
                        "alpha": [0.05],
                        "covariance": [[0.04]],
                    },
                    "policy": {"kind": "mlp"},
                    "evaluation": {
                        "states": 1,
                        "continuations": 2,
                        "continuation_batch": 2,
                        "z_outer_paths": 2,
                        "z_inner_paths": 1,
                    },
                },
                "jobs": [
                    {
                        "name": "learned",
                        "groups": ["paper"],
                        "checkpoint_alias": "checkpoint",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    suite = PaperSuite(manifest_path)
    assert suite.plan()[0]["missing_assets"] == ["checkpoint"]


def test_suite_resolves_relative_catalog_asset(tmp_path: Path) -> None:
    paper_dir = tmp_path / "paper"
    assets_dir = paper_dir / "assets"
    assets_dir.mkdir(parents=True)
    checkpoint = assets_dir / "policy.pt"
    checkpoint.write_bytes(b"placeholder")
    (paper_dir / "catalog.json").write_text(
        '{"checkpoint": "assets/policy.pt"}', encoding="utf-8"
    )
    config_dir = paper_dir / "configs"
    config_dir.mkdir()
    (config_dir / "learned.yaml").write_text(
        yaml.safe_dump(
            {
                "device": "cpu",
                "dtype": "float64",
                "model": {
                    "name": "merton",
                    "d": 1,
                    "T": 0.25,
                    "n_steps": 2,
                    "r": 0.03,
                    "gamma": 2.0,
                    "constraint": "unconstrained",
                    "alpha": [0.05],
                    "covariance": [[0.04]],
                },
                "policy": {"kind": "mlp"},
                "evaluation": {
                    "states": 1,
                    "continuations": 2,
                    "continuation_batch": 2,
                    "z_outer_paths": 2,
                    "z_inner_paths": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = paper_dir / "suite.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "output_root": "runs",
                "catalog": "paper/catalog.json",
                "jobs": [
                    {
                        "name": "learned",
                        "groups": ["paper"],
                        "config": "configs/learned.yaml",
                        "checkpoint_alias": "checkpoint",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    suite = PaperSuite(manifest_path)
    assert suite.plan()[0]["missing_assets"] == []
    assert suite.jobs[0].mapping["policy"]["checkpoint"] == str(checkpoint.resolve())
