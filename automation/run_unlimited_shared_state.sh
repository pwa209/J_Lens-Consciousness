#!/usr/bin/env bash
set -euo pipefail

ROOT="${JACACCESS_ROOT:-/root/autodl-tmp/jacobian-conscious-access}"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"
mkdir -p automation/state/full-study/locks logs/full-study/phase3
exec 9>"automation/state/full-study/locks/unlimited-shared-state.lock"
if ! flock -n 9; then
  echo "unlimited shared-state queue already owns the lock" >&2
  exit 75
fi

PYTHON="${JACACCESS_PYTHON:-$ROOT/.venv/bin/python}"
SNAKEMAKE="${JACACCESS_SNAKEMAKE:-$ROOT/.venv/bin/snakemake}"
summaries=()
analyses=()
for seed in $(seq 0 19); do
  summaries+=("results/machine/unlimited_shared_state/seed-${seed}/summary.json")
done

# Recompute all analysis tables because persistence was added after the first
# 80 runs; checkpoints are reused and no completed model is retrained.
architectures=(feedforward recurrent shared_workspace private_modules unlimited_shared_state)
for architecture in "${architectures[@]}"; do
  for seed in $(seq 0 19); do
    analyses+=("results/machine/${architecture}/seed-${seed}/intervention.json")
    analyses+=("results/machine/${architecture}/seed-${seed}/jacobian-signatures.parquet")
    analyses+=("results/machine/${architecture}/seed-${seed}/test-presence-by-bin.parquet")
  done
done

"$SNAKEMAKE" --snakefile workflows/Snakefile --cores 20 --resources gpu=4 mem_gb=90 \
  --rerun-incomplete --printshellcmds "${summaries[@]}"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$SNAKEMAKE" --snakefile workflows/Snakefile --cores 12 --resources gpu=1 mem_gb=90 \
  --rerun-incomplete --printshellcmds "${analyses[@]}"
"$SNAKEMAKE" --snakefile workflows/Snakefile --cores 20 --resources gpu=2 mem_gb=90 \
  --rerun-incomplete --printshellcmds \
  results/aggregate/machine/accuracy-matching/accuracy-matching.json \
  results/aggregate/machine/architecture-summary.csv
"$PYTHON" automation/check_machine_gate.py --mode production \
  --allow-scientific-failure \
  --output results/gates/machine-production-five-architectures.json
