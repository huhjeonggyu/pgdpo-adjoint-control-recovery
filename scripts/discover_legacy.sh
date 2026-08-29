#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

LEGACY_ROOT="${LEGACY_ROOT:-$HOME/legacy_mf_revision}"
args=(discover-legacy --root "$LEGACY_ROOT" --output paper/legacy_catalog.json)
if [[ "${LEGACY_STRICT:-0}" == "1" ]]; then
  args+=(--strict)
fi
run_mf "${args[@]}"
