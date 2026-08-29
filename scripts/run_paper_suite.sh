#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MANIFEST="${PAPER_MANIFEST:-paper/paper_suite.yaml}"
GROUP="${1:-paper}"
shift || true
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
run_mf suite --manifest "$MANIFEST" --group "$GROUP" "$@"
