#!/usr/bin/env bash
# End-to-end: assemble the CA DWR hybrid dataset, then train both models.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PY" ]; then
  echo "no venv at .venv — create it with:  python3.11 -m venv .venv && .venv/bin/pip install -e ." >&2
  exit 1
fi

"$PY" -m minesub pipeline "$@"
"$PY" -m minesub train --model both
echo
echo "Reports written to ./reports :"
ls -1 reports
