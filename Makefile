.PHONY: install test verify-market inspect smoke exact zeta \
        legacy-catalog paper-plan paper-smoke paper-exact \
        paper-section5 paper-appendix paper-full paper-collect \
        ridge-rerun audit-coauthor clean

PYTHON ?= $(if $(wildcard .venv/bin/python3),.venv/bin/python3,python3)
CLI = PYTHONPATH=src $(PYTHON) -m mf_revision.cli
LEGACY_ROOT ?= $(HOME)/legacy_mf_revision
GPU ?= 0

install:
	$(PYTHON) -m pip install --no-build-isolation -e '.[dev]'

test:
	$(PYTHON) -m compileall -q src tests scripts
	PYTHONPATH=src $(PYTHON) -m pytest -q

verify-market:
	PYTHONPATH=src $(PYTHON) scripts/verify_covariance_loading_consistency.py

inspect:
	$(CLI) inspect --config configs/merton_exact_smoke.yaml

smoke:
	bash scripts/run_smoke.sh

exact:
	bash scripts/run_exact.sh

zeta:
	bash scripts/run_zeta_revalidation.sh

legacy-catalog:
	LEGACY_ROOT="$(LEGACY_ROOT)" bash scripts/discover_legacy.sh

paper-plan:
	$(CLI) suite --manifest paper/paper_suite.yaml --group paper --plan

paper-smoke:
	CUDA_VISIBLE_DEVICES=$(GPU) CUBLAS_WORKSPACE_CONFIG=:4096:8 \
		bash scripts/run_paper_smoke.sh

paper-exact:
	CUDA_VISIBLE_DEVICES=$(GPU) CUBLAS_WORKSPACE_CONFIG=:4096:8 \
		bash scripts/run_paper_suite.sh exact

paper-section5:
	CUDA_VISIBLE_DEVICES=$(GPU) CUBLAS_WORKSPACE_CONFIG=:4096:8 \
		bash scripts/run_paper_suite.sh section5

paper-appendix:
	CUDA_VISIBLE_DEVICES=$(GPU) CUBLAS_WORKSPACE_CONFIG=:4096:8 \
		bash scripts/run_paper_suite.sh appendix

paper-full:
	CUDA_VISIBLE_DEVICES=$(GPU) CUBLAS_WORKSPACE_CONFIG=:4096:8 \
		bash scripts/run_paper_suite.sh paper

paper-collect:
	bash scripts/collect_paper.sh

ridge-rerun:
	CUDA_VISIBLE_DEVICES=$(GPU) CUBLAS_WORKSPACE_CONFIG=:4096:8 \
		bash scripts/rerun_ridge_consistent.sh

audit-coauthor:
	bash scripts/run_coauthor_experiment_checks.sh audit

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist htmlcov .coverage
	rm -rf runs runs_* paper/generated_configs paper/ridge_fix
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
