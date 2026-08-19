#!/usr/bin/env bash
set -euo pipefail

ROOT="${JACACCESS_ROOT:-/root/autodl-tmp/jacobian-conscious-access}"
STATE="$ROOT/automation/state/full-study"
LOGS="$ROOT/logs/full-study"
mkdir -p "$STATE" "$LOGS"
cd "$ROOT"

if [[ -f "$STATE/queue.pid" ]]; then
  existing=$(tr -dc '0-9' < "$STATE/queue.pid")
  if [[ -n "$existing" ]] && kill -0 "$existing" 2>/dev/null; then
    echo "Full-study queue is already running as PID $existing"
    exit 0
  fi
fi

source .venv/bin/activate
nohup setsid .venv/bin/python automation/autodl_queue.py --resume \
  >> "$LOGS/supervisor.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$STATE/queue.pid"
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "Queue failed during launch; inspect $LOGS/supervisor.log" >&2
  exit 1
fi
echo "Started full-study queue as PID $pid"
echo "Status: $STATE/queue.status.json"
