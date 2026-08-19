#!/usr/bin/env bash
set -euo pipefail

ROOT="${JACACCESS_ROOT:-/root/autodl-tmp/jacobian-conscious-access}"
STATE="$ROOT/automation/state/full-study"
pid_file="$STATE/queue.pid"

if [[ ! -f "$pid_file" ]]; then
  echo "No queue PID file exists"
  exit 0
fi
pid=$(tr -dc '0-9' < "$pid_file")
if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
  echo "Queue process is not running"
  exit 0
fi
command=$(tr '\0' ' ' < "/proc/$pid/cmdline")
if [[ "$command" != *"automation/autodl_queue.py"* ]]; then
  echo "Refusing to signal PID $pid because its command does not match the queue" >&2
  exit 2
fi
kill -TERM "$pid"
for _ in $(seq 1 30); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Queue stopped gracefully"
    exit 0
  fi
  sleep 1
done
echo "Queue is still stopping; no forced kill was issued" >&2
exit 1
