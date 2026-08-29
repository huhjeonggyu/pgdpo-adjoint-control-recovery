#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Legacy four-job convenience runner. The manifest-driven paper suite is preferred.
run_mf pipeline --config configs/affine_exact_canonical.yaml
run_mf pipeline --config configs/affine_exact_hedging.yaml
run_mf pipeline --config configs/merton_short_paper.yaml
run_mf pipeline --config configs/factor_constrained_paper.yaml
