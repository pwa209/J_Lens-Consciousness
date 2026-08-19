#!/usr/bin/env bash
set -euo pipefail

ROOT="${JACACCESS_ROOT:-/root/autodl-tmp/jacobian-conscious-access}"
cd "$ROOT"
mkdir -p automation/state/full-study/locks logs/full-study
exec 9>automation/state/full-study/locks/overnight-watchdog.lock
if ! flock -n 9; then
  echo "an overnight watchdog is already running" >&2
  exit 75
fi

export PATH="$ROOT/.venv/bin:/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="$ROOT/src"
failures=0

while true; do
  if grep -q '"status": "PASS"' automation/state/full-study/queue.status.json; then
    echo "$(date -u +%FT%TZ) study queue completed"
    exit 0
  fi

  if pgrep -f '[a]utomation/autodl_queue.py --resume' >/dev/null; then
    failures=0
  else
    failures=$((failures + 1))
    echo "$(date -u +%FT%TZ) restarting supervisor (attempt ${failures})"
    .venv/bin/python automation/autodl_queue.py --resume \
      >>logs/full-study/supervisor.log 2>&1 &
    if (( failures >= 3 )); then
      echo "$(date -u +%FT%TZ) supervisor failed three consecutive restarts" >&2
      exit 1
    fi
    sleep 120
    continue
  fi

  machine_summaries="$(find results/machine -name summary.json | wc -l)"
  if (( machine_summaries < 80 )) \
    && grep -q '"status": "RUNNING"' automation/state/full-study/phase2.status.json \
    && ! pgrep -f '[b]ackfill_machine_during_phase2.sh' >/dev/null; then
    echo "$(date -u +%FT%TZ) restarting guarded machine backfill"
    bash automation/backfill_machine_during_phase2.sh \
      >>logs/full-study/phase3/machine-backfill.log 2>&1 &
  fi
  sleep 60
done
