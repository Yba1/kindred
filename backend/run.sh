#!/usr/bin/env bash
# Boot the Kindred backend. Creates a venv on first run.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install -q --upgrade pip
  ./.venv/bin/pip install -q -r requirements.txt
fi

exec ./.venv/bin/uvicorn app.main:app --reload \
  --host "${KINDRED_HOST:-0.0.0.0}" --port "${KINDRED_PORT:-8000}"
