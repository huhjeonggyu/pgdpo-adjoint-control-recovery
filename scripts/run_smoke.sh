#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_mf pipeline --config configs/merton_exact_smoke.yaml
run_mf pipeline --config configs/affine_exact_smoke.yaml
run_mf pipeline --config configs/merton_short_smoke.yaml
run_mf pipeline --config configs/merton_cap_smoke.yaml
run_mf pipeline --config configs/factor_constrained_smoke.yaml
