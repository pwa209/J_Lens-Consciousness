#!/usr/bin/env bash
set -euo pipefail

ROOT="${JACACCESS_ROOT:-/root/autodl-tmp/jacobian-conscious-access}"
SEED="${1:?usage: run_machine_seed.sh SEED}"
cd "$ROOT"

mkdir -p automation/state/full-study/locks logs/full-study/phase3
exec 9>"automation/state/full-study/locks/machine-seed-${SEED}.lock"
if ! flock -n 9; then
  echo "seed ${SEED} is already running" >&2
  exit 75
fi

CACHE="data/derivatives/machine-stimuli/seed-${SEED}"
if [[ ! -s "$CACHE/manifest.json" ]]; then
  .venv/bin/python -m jacaccess.machine.cache \
    --seed "$SEED" \
    --config configs/models/machine.yaml \
    --output "$CACHE" \
    --workers "${JACACCESS_CACHE_WORKERS:-12}"
fi

architectures=(feedforward recurrent shared_workspace private_modules unlimited_shared_state)
pids=()
names=()
for architecture in "${architectures[@]}"; do
  output="results/machine/${architecture}/seed-${SEED}"
  if [[ -s "$output/summary.json" ]]; then
    echo "keeping completed ${architecture} seed ${SEED}"
    continue
  fi
  mkdir -p "$output"
  .venv/bin/python -m jacaccess.machine.train \
    --architecture "$architecture" \
    --seed "$SEED" \
    --config configs/models/machine.yaml \
    --stimulus-cache "$CACHE" \
    --output "$output" \
    >"logs/full-study/phase3/machine-${architecture}-seed-${SEED}.log" 2>&1 &
  pids+=("$!")
  names+=("$architecture")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "${names[$index]} seed ${SEED} failed" >&2
    failed=1
  fi
done
exit "$failed"
