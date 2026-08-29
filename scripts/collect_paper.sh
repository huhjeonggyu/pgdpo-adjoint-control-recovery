#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MANIFEST="${PAPER_MANIFEST:-paper/paper_suite.yaml}"
OUTPUT="${PAPER_COLLECTED_DIR:-paper/collected_full_shift}"
run_mf collect --manifest "$MANIFEST" --output "$OUTPUT"
