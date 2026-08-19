#!/usr/bin/env bash
set -euo pipefail

ROOT="${JACACCESS_ROOT:-/root/autodl-tmp/jacobian-conscious-access}"
STATE="$ROOT/automation/state/full-study"
cd "$ROOT"

if [[ -f "$STATE/queue.pid" ]]; then
  pid=$(tr -dc '0-9' < "$STATE/queue.pid")
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "process=RUNNING pid=$pid"
  else
    echo "process=NOT_RUNNING recorded_pid=${pid:-missing}"
  fi
else
  echo "process=NOT_STARTED"
fi

if [[ -f "$STATE/queue.status.json" ]]; then
  cat "$STATE/queue.status.json"
fi
for phase in phase0 phase1 phase2 phase3 phase4; do
  if [[ -f "$STATE/$phase.status.json" ]]; then
    echo "--- $phase ---"
    cat "$STATE/$phase.status.json"
  fi
done
