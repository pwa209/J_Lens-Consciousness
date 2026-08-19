#!/usr/bin/env bash
set -euo pipefail

ROOT="${JACACCESS_ROOT:-/root/autodl-tmp/jacobian-conscious-access}"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
mkdir -p logs/ordinary-study automation/state/ordinary-study
exec "$ROOT/.venv/bin/python" automation/ordinary_study_queue.py

