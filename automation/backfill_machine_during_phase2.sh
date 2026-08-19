#!/usr/bin/env bash
set -euo pipefail

ROOT="${JACACCESS_ROOT:-/root/autodl-tmp/jacobian-conscious-access}"
cd "$ROOT"
mkdir -p automation/state/full-study/locks logs/full-study/phase3

phase2_running() {
  grep -q '"status": "RUNNING"' automation/state/full-study/phase2.status.json
}

memory_below_guard() {
  local current inactive working_set
  current="$(cat /sys/fs/cgroup/memory.current)"
  inactive="$(awk '$1 == "inactive_file" {print $2}' /sys/fs/cgroup/memory.stat)"
  working_set=$(( current - inactive ))
  (( working_set < 60 * 1024 * 1024 * 1024 ))
}

for seed in $(seq 2 19); do
  lock="automation/state/full-study/locks/machine-seed-${seed}.lock"
  while phase2_running && ! flock -n "$lock" true; do
    sleep 30
  done
  if ! phase2_running; then
    echo "Phase 2 ended; leaving remaining seeds to the supervised Phase 3 queue"
    exit 0
  fi
  while pgrep -f 'python -m jacaccess.human_pipeline' >/dev/null || ! memory_below_guard; do
    if ! phase2_running; then
      echo "Phase 2 ended while waiting; handing off to Phase 3"
      exit 0
    fi
    sleep 30
  done
  bash automation/run_machine_seed.sh "$seed"
done
