#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_mf pipeline --config configs/merton_exact_smoke.yaml
run_mf pipeline --config configs/merton_cap_exact_smoke.yaml
run_mf pipeline --config configs/affine_exact_canonical.yaml
run_mf pipeline --config configs/affine_exact_hedging.yaml
