#!/usr/bin/env bash
# Shared source-tree launcher. This keeps scripts usable before editable install.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/.venv/bin/python3" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python3"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
MF_CMD=("$PYTHON_BIN" -m mf_revision.cli)

run_mf() {
  "${MF_CMD[@]}" "$@"
}
