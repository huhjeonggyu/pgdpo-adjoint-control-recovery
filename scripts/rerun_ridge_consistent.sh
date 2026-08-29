#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

"$PYTHON_BIN" scripts/verify_covariance_loading_consistency.py
"$PYTHON_BIN" -m pytest -q

"$PYTHON_BIN" scripts/prepare_ridge_consistent_suite.py \
  --manifest paper/paper_suite.yaml \
  --output paper/ridge_fix/paper_suite.yaml \
  --runs-root runs_ridge_consistent

run_mf suite \
  --manifest paper/ridge_fix/paper_suite.yaml \
  --group ridge_fix \
  --plan

run_mf suite \
  --manifest paper/ridge_fix/paper_suite.yaml \
  --group ridge_fix \
  --keep-going

run_mf collect \
  --manifest paper/ridge_fix/paper_suite.yaml \
  --output paper/collected_ridge_consistent

printf '\nCorrected runs:   %s\n' "$ROOT_DIR/runs_ridge_consistent"
printf 'Collected tables: %s\n' "$ROOT_DIR/paper/collected_ridge_consistent"
