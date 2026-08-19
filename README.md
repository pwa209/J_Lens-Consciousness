# Jacobian geometry of conscious access

This repository implements an ordinary observational and computational study of
Jacobian accessibility geometry in human EEG and controlled artificial
networks. It is derived from the Stage 1 Registered Report package dated
2026-07-30, but **does not claim Registered Report status** and does not require
acceptance in principle before target-data access.

The original safeguards against circular analysis remain useful:

- preprocessing and state-model fitting do not use awareness labels;
- learned transforms are fit within training folds;
- primary Jacobian metrics are constructed before condition contrasts;
- data, configuration, software and output hashes are recorded;
- exploratory analyses are kept separate from prespecified primary analyses.

## Current status

Production computation completed on 2026-08-16/17. All five ordinary-study
phases passed:

- 173 included human participants and 865/865 cross-fitted human folds;
- five machine architectures, 20 paired seeds, and 100/100 completed runs;
- five primary frequentist contrasts and directional Bayes factors;
- nested cross-validated incremental prediction;
- cross-dataset replication and five-theory human-machine comparison;
- six main and three supplementary publication figures;
- final result inventory and source-provenance freeze;
- a completed human-guided adaptation stage with 1,000 outer-fold runs and
  prespecified all-runs, gate-compliant, and complete-case retention analyses;
- six integrated native-R main figures in PNG, PDF, and SVG.

The human model-QC gate passed. Eight Kronemer folds (1.38% of that dataset's
folds) are retained as explicit warnings because their learned dynamics did not
clear the 2% persistence-improvement threshold; the prespecified dataset ceiling
was 20%. No raw EEG or participant-level result table is distributed through
GitHub.

Lightweight, disclosure-safe figures and aggregate outputs are provided in
`publication-results/`. The complete 5.1 GB reproducibility snapshot is kept as
a checksummed offline archive because it contains model checkpoints and
participant-level derivative tables that are inappropriate for ordinary Git.

## Human-guided adaptation and causal-testing stage

A linked analytical stage tests spontaneous architectural similarity, task-induced
convergence, and human-neural adaptability. It starts from the same 100 frozen
machine seed lineages and uses five outer Gabor participant folds with a matched
human-target/sham design. Adaptation uses only a low-level temporal EEG
representational-similarity distribution; all four final Jacobian geometry
metrics remain held-out evaluation endpoints.

Extension source, configuration, tests, and output documentation live under
`src/jacaccess/machine/human_adaptation.py`,
`configs/analysis/human_adaptation.yaml`, and
`results-extension/human-adaptation/`. Baseline characterization result files
are hashed before adaptation and verified again at finalization.

The adaptation results constrain the article's claim: no architecture produced
a Holm-corrected held-out geometry improvement, and the nominal
capacity-limited sham contrast was not robust to multiplicity correction or the
complete-case sensitivity analysis. The result is therefore a falsification and
boundary-condition test, not a positive workspace or fine-tuning claim.

See [`docs/RESULTS_AND_REPRODUCIBILITY.md`](docs/RESULTS_AND_REPRODUCIBILITY.md)
for result locations and reproduction commands. Historical plans and deviations
are retained in [`ROADMAP.md`](ROADMAP.md) and [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Local checks

The core reference implementation only needs NumPy:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_local_checks.py
python -m jacaccess.synthetic --output results/synthetic
```

Install the full development environment and run the test suite with:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Create the upload archive with:

```powershell
python scripts/package_for_autodl.py
```

The complete environment is created on AutoDL using:

```bash
bash environment/bootstrap_autodl.sh
```

The bootstrap finishes by running:

```bash
snakemake --snakefile workflows/Snakefile --cores 25 \
  --resources gpu=1 results/stage2_ready.flag
```

## Data policy

Raw target data are intentionally absent from this repository. Place immutable
source archives under `data/raw/` only on the execution host. Commit manifests
and checksums, never participant recordings.

## License

Code is released under the [MIT License](LICENSE). Dataset licences and access
conditions remain those of the original data providers and are not superseded
by this repository.
