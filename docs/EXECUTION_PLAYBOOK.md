# Execution playbook

This is the shortest safe path from the source archive to the complete study.
It deliberately separates code verification from scientific source mapping.

## 1. Bootstrap and verify the GPU host

```bash
unzip jacobian-conscious-access-upload.zip
cd jacobian-conscious-access
bash environment/bootstrap_autodl.sh
source .venv/bin/activate
```

The bootstrap runs `results/stage2_ready.flag`. Do not proceed if CUDA tests,
machine smoke tests, available RAM, VRAM, or scratch checks fail.

## 2. Acquire pilot data only

Use:

```bash
bash scripts/download_datasets.sh gabor
bash scripts/download_datasets.sh somato
NITRC_COOKIE='reviewed-session-cookie' bash scripts/download_datasets.sh kronemer
```

When the repository interface permits it, select one participant before a full
download. The commands are resumable where the source protocol supports ranges.
Each completed raw tree gets a SHA-256 manifest under `data/manifests/`.

## 3. Inspect without converting

For one participant from each source:

```bash
python -m jacaccess.io.standardize inspect \
  --dataset gabor --participant PARTICIPANT \
  --raw-root data/raw/gabor \
  --output results/source-inspection/gabor-PARTICIPANT.json
```

Repeat for `kronemer` and `somato`. Reconcile signal file count, sampling rate,
channel names, event columns, units, trial counts, and condition encodings
against the repository documentation.

## 4. Fill the verification-only configuration fields

Do not copy these as guessed values. Populate each dataset YAML from inspection:

```yaml
adapter_status: verified
event_columns:
  original_trial_id: SOURCE_COLUMN
  onset_seconds: SOURCE_COLUMN
  event_type: SOURCE_COLUMN
  # plus every required physical and condition field
output_channel_groups:
  posterior: [REVIEWED_ZERO_BASED_INDICES]
  frontoparietal: [REVIEWED_ZERO_BASED_INDICES]
primary_contrasts:
  - condition_field: VERIFIED_COMMON_FIELD
    positive: VERIFIED_LEVEL
    negative: VERIFIED_LEVEL
    window_ms: [START, STOP]
prediction_outcome: VERIFIED_BINARY_FIELD
```

The somatosensory matrix adapter infers conditions from folder names. Confirm
those folder semantics manually before setting `adapter_status: verified`.

Add reviewed participant rows to `configs/execution/participants.tsv`:

```text
dataset_id	participant_id	include	reason
gabor	sub-XX	true	pilot verified
```

Then run:

```bash
python scripts/check_adapter_gate.py
```

## 5. Pilot one participant end to end

```bash
snakemake --snakefile workflows/Snakefile --cores 25 \
  --resources gpu=1 --rerun-incomplete --printshellcmds \
  results/human/gabor/PARTICIPANT/fold-0/summary.json
```

Review:

- preprocessing `qc.json` and artifact distributions;
- accepted trial/event reconciliation;
- PCA rank and output readout definitions;
- validation loss and held-out persistence improvement;
- Parquet schema, hashes, time axis, and missing-value patterns;
- confirmation that `condition_joined_after_metric_seal` is true.

Repeat the pilot for the other two datasets and all five folds before expanding
the participant table.

## 6. Launch production

```bash
snakemake --snakefile workflows/Snakefile --cores 25 \
  --resources gpu=1 --rerun-incomplete --keep-going --printshellcmds \
  results/study_complete.flag
```

This target runs human analysis, 80 machine fits, causal interventions, strict
aggregation, nested prediction, primary sign-flip/cluster tests, directional
Bayes factors, and figures.

All long fits write resumable checkpoints. Restart the same command after host
or network interruption. Do not remove `.snakemake`, checkpoints, raw
manifests, or sealed metric indices until verified backups exist.

## 7. Back up and audit

Back up at minimum:

- source archive and its SHA-256 sidecar;
- reviewed YAML/TSV configuration;
- raw manifests;
- participant QC and exclusion records;
- human partition indices and Parquet parts;
- machine final checkpoints and intervention records;
- aggregate tables, statistics, figures, environment report, and `pip-freeze`.

The raw data themselves remain governed by each source repository's terms.
