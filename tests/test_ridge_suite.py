from __future__ import annotations

from mf_revision.experiments.suite import PaperSuite
from scripts.prepare_ridge_consistent_suite import affected_job_names


def test_corrected_suite_includes_changed_graph_comparison() -> None:
    suite = PaperSuite("paper/paper_suite.yaml")
    selected = affected_job_names(suite)

    assert "t12_lko_short_d100" in selected
    assert "app_graph_lko_d100_cl" in selected
    assert "app_graph_lko_d100_compare" in selected
    assert "t5_budget_seed1_e500" not in selected
    assert len(selected) == 28
