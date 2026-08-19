# AutoDL handoff

## Host selection

Minimum recommended production host:

- one RTX 5090 with 32 GB VRAM;
- 25 or more CPU cores;
- 128 GB RAM minimum, 256 GB preferred;
- 3 TB local NVMe scratch minimum;
- reliable external/object storage for raw archives and final outputs.

Do not select a host based on GPU alone. PyPREP, filtering and ICA are commonly
limited by RAM and disk throughput.

## Upload

Upload only the repository source archive. Do not upload local caches,
`.venv`, raw data or results.

```bash
cd /root/autodl-tmp
unzip jacobian-conscious-access.zip
cd jacobian-conscious-access
bash environment/bootstrap_autodl.sh
source .venv/bin/activate
python -m jacaccess.environment_check --output environment/environment-report.json
```

The environment check must report:

- CUDA available;
- GPU name containing `RTX 5090`;
- compute capability 12.0 or later;
- at least 30 GiB visible VRAM;
- at least 100 GiB system RAM;
- at least 2.5 TiB free scratch before raw-data extraction.

## Storage layout

Use `/root/autodl-tmp` for high-throughput working data:

```text
/root/autodl-tmp/jacobian-conscious-access/
  data/raw/          immutable downloaded archives and extracted source files
  data/derivatives/  replaceable preprocessing caches
  results/           metrics, statistics, figures and provenance
```

Back up raw manifests, sealed metrics, statistics and logs to reliable object
storage. AutoDL local disks are scratch storage, not the only archive.

## Data acquisition sequence

1. Download one participant from each repository first. The acquisition entry
   points are `bash scripts/download_datasets.sh gabor`, `somato`, or
   `kronemer`; restrict the source selection to a pilot participant before
   invoking them when the repository supports selective download.
2. Generate SHA-256 manifests before extraction.
3. Inspect source events and channel metadata with
   `python -m jacaccess.io.standardize inspect`.
4. Run dataset-adapter and preprocessing smoke tests.
5. Record real download, extraction, RAM and preprocessing measurements.
6. Only then download and process the complete selected sample.

## Execution sequence

The bootstrap runs the complete Stage 2 readiness target. It can also be
repeated explicitly:

```bash
snakemake --snakefile workflows/Snakefile --cores 25 \
  --resources gpu=1 --rerun-incomplete --printshellcmds \
  results/stage2_ready.flag
```

This validates:

- all configuration contracts;
- five synthetic human folds;
- human analytic Jacobians, losses and checkpoint recovery;
- parameter counts and tensor contracts for all four machine systems;
- small forward passes for two seeds per architecture;
- exact machine future-logit Jacobians.

Production participant/fold targets will be enabled after the three dataset
adapters pass:

```bash
snakemake --snakefile workflows/Snakefile --cores 25 \
  --resources gpu=1 --rerun-incomplete --printshellcmds \
  results/study_complete.flag
```

Use `tmux`, `screen`, or a supervised batch mechanism. Never rely on an open
browser tab to keep a multi-day process alive.

Before the production command, each dataset YAML must contain:

- `adapter_status: verified`;
- verified `event_columns` (the somatosensory matrix adapter instead requires
  manual confirmation of its inferred folder semantics);
- at least two `output_channel_groups`;
- one or more explicit `primary_contrasts`;
- a reviewed row for every included participant in
  `configs/execution/participants.tsv`.
