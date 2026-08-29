#!/usr/bin/env bash
set -euo pipefail

# Targeted source/result audits and optional isolated reruns.
#
# Usage:
#   bash scripts/run_coauthor_experiment_checks.sh audit
#   GPU=0 bash scripts/run_coauthor_experiment_checks.sh table6-exact
#   GPU=0 bash scripts/run_coauthor_experiment_checks.sh table6-all
#   GPU=0 bash scripts/run_coauthor_experiment_checks.sh lko-budget
#
# Reruns are written under runs_coauthor_rerun by default and never overwrite
# the manuscript run tree.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODE="${1:-audit}"
GPU="${GPU:-0}"
RUN_ROOT="${RUN_ROOT:-runs_paper_full_shift}"
RERUN_ROOT="${RERUN_ROOT:-runs_coauthor_rerun}"
FORCE="${FORCE:-0}"
MANIFEST="${PAPER_MANIFEST:-paper/paper_suite.yaml}"

mkdir -p paper/coauthor_checks logs "$RERUN_ROOT"

run_audit() {
  local run_root="${1:-$RUN_ROOT}"
  local output="${2:-paper/coauthor_checks}"
  "$PYTHON_BIN" scripts/coauthor_experiment_audit.py \
    --repo "$ROOT_DIR" \
    --run-root "$run_root" \
    --output "$output" \
    --device auto \
    --strict
}

materialize_group() {
  local group="$1"
  run_mf suite --manifest "$MANIFEST" --group "$group" --materialize >/dev/null
}

run_config() {
  local config="$1"
  local name
  name="$($PYTHON_BIN - "$config" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = yaml.safe_load(handle)
print(payload.get("name") or __import__("pathlib").Path(sys.argv[1]).stem)
PY
)"
  local final="$RERUN_ROOT/$name/recovery_summary.json"
  if [[ -f "$final" && "$FORCE" != "1" ]]; then
    echo "[skip complete] $name ($final)"
    return 0
  fi
  echo "[run] $name -> $RERUN_ROOT/$name"
  CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    run_mf pipeline --config "$config" --output-root "$RERUN_ROOT"
}

case "$MODE" in
  audit)
    run_audit "$RUN_ROOT" paper/coauthor_checks
    ;;
  table6-exact)
    materialize_group exact
    run_config paper/generated_configs/t6_merton_cap_d100_exact.yaml
    "$PYTHON_BIN" scripts/coauthor_experiment_audit.py \
      --repo "$ROOT_DIR" --run-root "$RERUN_ROOT" \
      --output paper/coauthor_checks_rerun --device auto \
      --allow-missing-runs
    ;;
  table6-all)
    materialize_group table6
    for config in \
      paper/generated_configs/t6_merton_cap_d2.yaml \
      paper/generated_configs/t6_merton_cap_d10.yaml \
      paper/generated_configs/t6_merton_cap_d100.yaml \
      paper/generated_configs/t6_merton_cap_d100_exact.yaml; do
      run_config "$config"
    done
    run_audit "$RERUN_ROOT" paper/coauthor_checks_rerun
    ;;
  lko-budget)
    materialize_group table12
    for config in \
      paper/generated_configs/t12_lko_short_d10.yaml \
      paper/generated_configs/t12_lko_short_d50.yaml \
      paper/generated_configs/t12_lko_short_d100.yaml; do
      run_config "$config"
    done
    "$PYTHON_BIN" scripts/coauthor_experiment_audit.py \
      --repo "$ROOT_DIR" --run-root "$RERUN_ROOT" \
      --output paper/coauthor_checks_rerun --device auto \
      --allow-missing-runs
    ;;
  all-targets)
    materialize_group table6
    materialize_group table8
    materialize_group table12
    for config in \
      paper/generated_configs/t6_merton_cap_d2.yaml \
      paper/generated_configs/t6_merton_cap_d10.yaml \
      paper/generated_configs/t6_merton_cap_d100.yaml \
      paper/generated_configs/t6_merton_cap_d100_exact.yaml \
      paper/generated_configs/t8_merton_cap_iid.yaml \
      paper/generated_configs/t8_merton_cap_wealth_edge.yaml \
      paper/generated_configs/t8_merton_cap_stress.yaml \
      paper/generated_configs/t8_lko_cap_iid.yaml \
      paper/generated_configs/t8_lko_cap_wealth_edge.yaml \
      paper/generated_configs/t8_lko_cap_factor_edge.yaml \
      paper/generated_configs/t8_lko_cap_stress.yaml \
      paper/generated_configs/t12_lko_short_d10.yaml \
      paper/generated_configs/t12_lko_short_d50.yaml \
      paper/generated_configs/t12_lko_short_d100.yaml; do
      run_config "$config"
    done
    run_audit "$RERUN_ROOT" paper/coauthor_checks_rerun
    ;;
  *)
    cat >&2 <<EOF_USAGE
Unknown mode: $MODE

Valid modes:
  audit          audit an existing run tree; no simulation
  table6-exact   rerun only the analytical d=100 Table-6 pipeline
  table6-all     rerun all four Table-6 jobs
  lko-budget     rerun constrained-factor d=10,50,100 jobs
  all-targets    rerun all jobs used by the targeted audit

Environment variables:
  GPU=0                  physical GPU index (default: 0)
  RUN_ROOT=path          existing run tree for audit mode
  RERUN_ROOT=path        isolated rerun root (default: runs_coauthor_rerun)
  PAPER_MANIFEST=path    suite manifest
  FORCE=1                replace completed reruns
  PYTHON=/path/python3   Python executable
EOF_USAGE
    exit 2
    ;;
esac
