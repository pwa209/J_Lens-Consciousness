#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/jacobian-conscious-access
STATE="$ROOT/automation/state"
LOGS="$ROOT/logs/bootstrap"
PYTHON=/root/miniconda3/bin/python

mkdir -p "$STATE" "$LOGS"
cd "$ROOT"

fail() {
  code=$?
  printf '{"status":"FAILED","exit_code":%d,"finished_at_utc":"%s"}\n' \
    "$code" "$(date -u +%FT%TZ)" > "$STATE/bootstrap.failed.json"
  exit "$code"
}
trap fail ERR

printf '{"status":"RUNNING","started_at_utc":"%s"}\n' \
  "$(date -u +%FT%TZ)" > "$STATE/bootstrap.status.json"

if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON" -m venv --system-site-packages .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[eeg,workflow,dev]'
python -m pip install 'awscli>=1.36,<2'

python -m jacaccess.environment_check \
  --scratch /root/autodl-tmp \
  --allow-development-host \
  --output environment/environment-report.json
python -m pip freeze > environment/pip-freeze.txt
python -m compileall -q src scripts
python -m unittest discover -s tests -p 'test_*.py' -v \
  2>&1 | tee "$LOGS/unittest.log"

snakemake --snakefile workflows/Snakefile --cores 25 \
  --resources gpu=1 --rerun-incomplete --printshellcmds \
  results/local_ready.flag 2>&1 | tee "$LOGS/local-ready.log"

for architecture in feedforward recurrent shared_workspace private_modules; do
  for seed in 0 1; do
    python -m jacaccess.machine.smoke \
      --architecture "$architecture" \
      --seed "$seed" \
      --output "results/machine-smoke/$architecture/seed-$seed.json"
  done
done

printf '{"status":"PASS","finished_at_utc":"%s"}\n' \
  "$(date -u +%FT%TZ)" > "$STATE/bootstrap.status.json"
touch "$STATE/BOOTSTRAP_COMPLETE"
