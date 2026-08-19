# Results and reproducibility

## Completed production scope

- Human arm: 173 included participants and 865/865 cross-fitted folds.
- Machine arm: five architectures, 20 paired seeds, and 100/100 runs.
- Inference: five frequentist contrast reports, five directional Bayes factors,
  cross-dataset replication, capacity-control equivalence tests, and nested
  incremental prediction.
- Human-guided adaptation: 1,000 outer-fold runs (five architectures, 20 seeds,
  five folds, human target and matched sham), with 34 retention-gate failures
  retained rather than discarded.
- Reporting: six native-R main figures in PNG, PDF, and SVG, plus the existing
  supplementary QC outputs.

The final nested-prediction estimate was an incremental AUC of approximately
0.00008 (directional sign-flip p = 0.501). The feedforward architecture had the
smallest mean human-machine geometry distance in the specified theory
comparison. These are computational results, not claims of causal equivalence
between human consciousness and artificial systems.

No architecture showed a Holm-corrected held-out adaptation benefit. Mean
human-adapted improvement relative to the task-trained stage was -0.00747 for
feedforward, -0.10875 for recurrent, +0.01517 for private modules, -0.07888 for
capacity-limited shared state, and -0.09528 for unlimited shared state; every
Holm-adjusted p value was 1. The 34 retention failures were exactly balanced
between human-target and sham adaptation (17 each). All-runs, gate-compliant,
and complete-case sensitivity analyses yielded no Holm-corrected positive
effect. The nominal capacity-limited sham contrast in the all-runs analysis
(p = 0.0281) did not survive multiplicity correction (Holm p = 0.1405) and was
weaker in the complete-case analysis (p = 0.0587; Holm p = 0.293).

## Public results

`publication-results/` contains only lightweight aggregate artifacts suitable
for a public source repository:

- article digest and figure manifest;
- publication figures;
- frequentist JSON reports and directional Bayes-factor JSON reports;
- aggregate prediction, theory-comparison, gate, and provenance reports;
- aggregate machine summaries.
- the six native-R production figures and their integrity manifest.

Participant-level contrast CSVs, human trial-time tables, checkpoints, raw EEG,
and preprocessed EEG are excluded from GitHub.

## Native-R figure reproduction

The publication figure stage is now:

```bash
python scripts/prepare_science_advances_figure_data.py \
  --project-root . \
  --extension-root results-extension/human-adaptation \
  --output results-extension/human-adaptation/figures/science-advances-r/data
Rscript --vanilla scripts/render_science_advances_figures.R \
  --data results-extension/human-adaptation/figures/science-advances-r/data \
  --output results-extension/human-adaptation/figures/science-advances-r/final \
  --manifest results-extension/human-adaptation/figures/science-advances-r/final/manifest.json
```

Production used R 4.1.2, ggplot2 3.3.5, patchwork 1.1.1, and svglite 2.1.0.
The public repository contains the rendered files but not participant-level
source tables; those remain in the checksummed offline results package.

## Offline complete snapshot

The complete SSD snapshot is stored as
`jacobian-conscious-access-complete-20260817.tar` with a neighbouring SHA-256
checksum file. It contains the full `results/` tree, logs, code, configurations,
workflow definitions, tests, and provenance. It deliberately excludes raw
source EEG, preprocessed EEG, the Python virtual environment, and download
caches. Raw and preprocessed EEG remain governed by their original providers.

Verify the archive before extraction:

```bash
sha256sum -c jacobian-conscious-access-complete-20260817.tar.sha256
tar -xf jacobian-conscious-access-complete-20260817.tar
```

## Reproduction

Create the environment and validate the code:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,eeg,workflow]"
python -m pytest -q
```

With provider-authorized data placed under `data/raw/`, execute the resumable
ordinary-study supervisor on the configured AutoDL host:

```bash
bash environment/bootstrap_autodl.sh
bash automation/run_ordinary_study.sh
```

The final state is recorded under `automation/state/ordinary-study/`, and the
completion inventory is written to
`results/provenance/final-results-inventory.json`.
