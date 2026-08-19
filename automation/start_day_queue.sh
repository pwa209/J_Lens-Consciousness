#!/usr/bin/env bash
set -euo pipefail

ROOT="${JACACCESS_ROOT:-/root/autodl-tmp/jacobian-conscious-access}"
cd "$ROOT"
mkdir -p logs/full-study automation/state/full-study/locks

bash automation/start_autodl_queue.sh
if ! pgrep -f '[o]vernight_watchdog.sh' >/dev/null; then
  nohup setsid bash automation/overnight_watchdog.sh \
    >>logs/full-study/overnight-watchdog.log 2>&1 < /dev/null &
  echo "Started full-day watchdog as PID $!"
else
  echo "Full-day watchdog is already running"
fi

sleep 3
pgrep -af 'autodl_queue.py --resume|overnight_watchdog.sh' || true
cat automation/state/full-study/queue.status.json
