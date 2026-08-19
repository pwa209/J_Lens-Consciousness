#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This bootstrap is intended for the Linux AutoDL host." >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
python -m pip install -e ".[eeg,workflow,dev]"

python -m jacaccess.environment_check \
  --scratch /root/autodl-tmp \
  --output environment/environment-report.json

python -m pip freeze > environment/pip-freeze.txt
snakemake --snakefile workflows/Snakefile \
  --cores 25 \
  --resources gpu=1 \
  --rerun-incomplete \
  --printshellcmds \
  results/stage2_ready.flag

echo "AutoDL Stage 2 environment and source checks are ready."
