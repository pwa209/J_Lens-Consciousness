# Full AutoDL study roadmap

## 1. Aim and operating principle

This is an ordinary empirical study, not a Registered Report. The analysis
plan is nevertheless frozen before primary outcomes are inspected so that
exploratory flexibility does not leak into the confirmatory tests.

All data acquisition, source inspection, preprocessing, model fitting,
Jacobian analysis, machine experiments, inference, figures, and archival
packaging will run on the AutoDL host. The local project remains a source-code
copy and recovery point, not a compute dependency.

## 2. Current starting state

As of 2026-08-09 the active AutoDL allocation has:

- one RTX 5090 with approximately 32 GiB VRAM;
- 25 cgroup CPU cores;
- 90 GiB cgroup RAM;
- 3.2 TiB XFS scratch, about 3.0 TiB free after the existing projects;
- the project at `/root/autodl-tmp/jacobian-conscious-access`;
- a passing measured bootstrap, unit-test, and two-seed-per-architecture
  machine smoke run recorded in `automation/state/bootstrap.status.json`.

The 3 TiB storage gate is now satisfied. The remaining material capacity risk
is RAM: the analysis specification requests at least 128 GiB, while this host
actually exposes 90 GiB. Production is therefore conditional on measured
one-participant peak memory and serial preprocessing.

The source adapters are intentionally not yet verified. Dataset YAML event
mappings, output groups, primary contrasts, prediction outcomes, and the
participant table must be populated from real source files before production.

## 3. Fixed directory and retention policy

Use this layout throughout the study:

```text
/root/autodl-tmp/jacobian-conscious-access/
  data/downloads/       immutable archives and download receipts
  data/raw/             extracted source files used by adapters
  data/derivatives/     replaceable standardized/preprocessed data
  results/              folds, metrics, inference, figures, provenance
  logs/                 timestamped stage and resource logs
  automation/state/     atomic PASS, FAIL, RUNNING, and queue state
  backups/staging/      compact material awaiting off-host transfer
```

Retention rules:

1. Never alter source archives after hashing.
2. Do not keep duplicate extractions when a verified selective extraction is
   sufficient.
3. Do not delete an archive merely because extraction succeeded. It may be
   removed only after its checksum, source URL, retrieval date, and a verified
   off-host copy or documented public re-download path exist.
4. Derivatives may be recreated, but sealed metric tables, statistical
   outputs, logs, manifests, configuration snapshots, and checksums require an
   off-host backup.
5. Stop new downloads when free scratch falls below 500 GiB. The final 300 GiB
   is an untouchable recovery reserve.

### Provisional disk budget

| Material | Working allowance |
|---|---:|
| Existing AutoDL content | 215 GiB |
| Gabor source | 45 GiB |
| Somatosensory source and extraction | 100 GiB |
| Kronemer archives | 1,100 GiB |
| Selectively extracted EEG | 350 GiB |
| Human derivatives and caches | 500 GiB |
| Machine checkpoints and analyses | 300 GiB |
| Logs, tables, figures, staging | 100 GiB |
| Protected recovery reserve | 300 GiB |
| Total envelope | 3,010 GiB |

These are planning ceilings, not measured facts. Replace each allowance after
the pilot downloads. If the revised projection exceeds 2.7 TiB of new study
content, retain only necessary EEG members from Kronemer archives or attach
external storage before continuing.

## 4. Execution gates

| Gate | Evidence required | What it unlocks |
|---|---|---|
| G0 host ready | cgroup-aware CPU/RAM/GPU/disk report; bootstrap PASS | pilot acquisition |
| G1 source contracts | three inspected pilot participants; verified labels, channels, timing, counts | adapter verification |
| G2 human pilot | one participant per dataset completes preprocessing and five folds; RAM and runtime recorded | complete human acquisition |
| G3 human production | QC, leakage, completeness, and fold audits pass | primary human inference |
| G4 machine benchmark | two seeds per architecture pass; one timed production-scale run per architecture; parameter counts within 10% | all 80 machine runs |
| G5 machine production | accuracy matching passes; at least 15 valid paired seeds; all interventions auditable | cross-system comparison |
| G6 study seal | immutable metrics, config/code/input hashes, exclusions, statistics, and backup receipts | final figures and report |

Scientific gate failures are not auto-corrected. A queue may retry transient
download or process failures three times, but inconsistent event semantics,
failed leakage checks, invalid contrasts, or systematic model-QC failures stop
the queue for documented review.

## 5. Phase 0 - production hardening and freeze

Estimated elapsed time: 0.5-1 day. Current status: bootstrap passed; hardening
items remain.

1. Make environment reporting cgroup-aware. Record both host and effective
   cgroup CPU/RAM values; enforce decisions on the latter.
2. Freeze a source snapshot with Git commit ID or a source-tree SHA-256
   manifest. Save `pip freeze`, CUDA/PyTorch versions, GPU details, filesystem
   capacity, and configuration hashes under `results/provenance/`.
3. Add Snakemake resources to production rules: `gpu=1`, `mem_gb`,
   `download_slots=1`, and explicit threads. The queue default is:

   ```bash
   snakemake --snakefile workflows/Snakefile --cores 25 \
     --resources gpu=1 mem_gb=85 download_slots=1 \
     --rerun-incomplete --keep-going --printshellcmds
   ```

4. Limit BLAS/OpenMP fan-out so nested libraries do not silently exceed 25
   cores.
5. Use atomic state files written first as `RUNNING`, then renamed to `PASS` or
   `FAIL`; store the actual PID, command, start/end time, exit code, and last
   completed target.
6. Run a small checkpoint-and-resume test for one human fold and one machine
   run.
7. Prepare an off-host backup destination before any irreplaceable result is
   created.

Exit: G0 passes and the frozen code/config snapshot can reproduce the remote
bootstrap.

## 6. Phase 1 - acquisition, inspection, and adapter verification

Estimated elapsed time: 2-5 days, dominated by pilot retrieval and source
inspection.

### 6.1 Pilot acquisition order

1. **Gabor/OpenNeuro ds005273:** obtain sidecars plus one complete participant
   first. The full public dataset is approximately 44.4 GB and contains 33
   recordings, 63 EEG channels, and 1000 Hz data.
2. **Somatosensory OSF hqkym:** obtain source documentation and one participant
   represented in both relevant report/no-report structures. Inspect raw and
   preprocessed forms before selecting the authoritative input.
3. **Kronemer BMVP/NITRC:** retrieve one report participant archive and one
   no-report participant archive. Generate the complete download manifest from
   the public file list, but do not launch it until selective extraction and
   disk projections are verified.

Every transfer writes URL, remote filename, expected/observed size, retrieval
timestamp, SHA-256, retry count, and HTTP/result status to a machine-readable
manifest. Partial downloads use resumable transfers and a `.part` suffix.

### 6.2 Source-contract inspection

For each pilot:

1. Inventory archive members before extraction.
2. Extract only source documentation, event/channel sidecars, and the selected
   participant.
3. Run `python -m jacaccess.io.standardize inspect` and save its output.
4. Reconcile trial counts, event codes, sampling rate, channel names/types,
   reference, bad-channel metadata, and report/awareness fields against the
   source documentation.
5. Confirm the unit of every time field and the alignment of stimulus,
   response, and report markers.
6. Define the condition-blind signal input, target/output channel groups,
   primary contrasts, prediction outcome, exclusions, and missing-data rules.
7. Never infer scientific labels solely from folders or filenames. Ambiguous
   Somatosensory folder semantics require a source-backed decision recorded in
   `docs/DECISIONS.md`.
8. Populate the three dataset YAML files and add reviewed participants to
   `configs/execution/participants.tsv`.
9. Set `adapter_status: verified` only after an automated reconciliation test
   and a human-readable inspection report both pass.

### 6.3 Pilot preprocessing and resource benchmark

Run one participant per dataset serially. Wrap each step with `/usr/bin/time
-v`, sample `nvidia-smi`, and record scratch usage before and after. Measure:

- download and extraction time;
- peak resident RAM and GPU memory;
- preprocessing/ICA time;
- standardized and derivative size;
- retained trials/channels and every exclusion reason;
- five-fold fitting and Jacobian-analysis time;
- checkpoint/restart behavior.

Do not run two ICA or full-participant preprocessing jobs concurrently on the
90 GiB host. G2 passes only if every pilot stays below 80 GiB peak RSS, leaving
operating headroom. If any exceeds 80 GiB, reduce chunk sizes or use memmaps and
rerun; if it still exceeds the limit, upgrade RAM before production.

Exit: G1 and G2 pass, the production sample is enumerated, and the measured
disk/time projection fits the host.

## 7. Phase 2 - complete human analysis

Estimated elapsed time: 10-18 days after G2.

### 7.1 Full acquisition and extraction

1. Download Gabor and Somatosensory completely using verified manifests.
2. Download the selected Kronemer EEG archives in manifest order, with one
   active transfer and a disk check before each archive.
3. Hash every archive and inventory it before extraction.
4. Selectively extract only the required EEG, event, montage, and participant
   metadata members. Avoid simultaneous archive plus duplicate complete-tree
   copies where reliable re-download or off-host backup exists.
5. Recompute the disk projection after 10%, 25%, and 50% of Kronemer retrieval.

### 7.2 Preprocessing and QC

1. Standardize identifiers and events with verified adapters.
2. Run outcome-blind filtering, channel QC, rereferencing, artifact handling,
   epoching, and caching.
3. Process one participant at a time; allocate at most 20 CPU threads to the
   participant and retain 5 for downloads/monitoring.
4. Produce participant-level QC before joining awareness/report labels.
5. Apply only frozen inclusion and exclusion rules. Log both included and
   excluded participants/trials; never silently drop failures.
6. Seal a preprocessing manifest containing source, adapter, config, and output
   hashes.

### 7.3 Five-fold human models and Jacobian metrics

For every eligible participant:

1. Assign deterministic five-fold splits.
2. Fit PCA/whitening and readouts on training data only.
3. Fit residual dynamics with resumable checkpoints.
4. Verify analytic Jacobians against autograd on the prescribed sample.
5. Stream propagators in chunks; do not store unnecessary dense Jacobian
   tensors.
6. Write partitioned Parquet metric tables and a partition index.
7. Require held-out model QC for all folds before that participant contributes
   to primary aggregation.

GPU jobs run one at a time. A CPU-only preprocessing job may overlap one GPU
fold only after telemetry shows combined RSS below 80 GiB and disk I/O remains
stable. Otherwise the stages alternate.

### 7.4 Human sealing and inference

1. Audit fold duplication, PCA leakage, target leakage, missing partitions,
   hash mismatches, and configuration drift.
2. Freeze the model-QC and exclusion report.
3. Seal the condition-blind metric tables.
4. Join condition labels only after sealing.
5. Aggregate H1-H5 inputs, including nested H5 prediction features.
6. Run the declared frequentist, permutation, and directional Bayes analyses.
7. Preserve null, conflicting, and failed-QC results without changing the
   pipeline in response.

Exit: G3 passes and all human primary results can be regenerated from the
sealed tables without refitting models.

## 8. Phase 3 - complete machine experiment

Estimated elapsed time: 4-8 days after benchmarking; it can partially overlap
CPU-heavy Phase 2 work.

The existing two-seed-per-architecture CUDA smoke gate has passed. Before the
80-run queue, benchmark one production-scale seed for each of the four
architectures and record examples/second, peak VRAM/RAM, checkpoint size, and
analysis time. Revise the schedule from those measurements.

### 8.1 Production queue

1. Freeze shared task generation, training schedule, optimizer, stopping
   rules, seeds 0-19, and the common 32-D bottleneck contract.
2. Verify architecture parameter counts differ by no more than 10%.
3. Train four architectures across 20 paired seeds: 80 total runs.
4. Generate image batches deterministically on demand rather than persisting a
   large duplicate corpus.
5. Use mixed precision, one GPU run at a time, periodic atomic checkpoints,
   and resume from the latest valid checkpoint.
6. Retry infrastructure failures up to three times. Mark numerical or model-QC
   failures as scientific failures and retain their artifacts.

### 8.2 Machine analysis

For each completed run:

1. Apply the frozen threshold-bin accuracy-matching rule.
2. Compute exact current-to-future logit Jacobians at the six specified steps.
3. Derive the same geometry signatures used for the human analysis.
4. Run top-four and random-subspace interventions with paired randomness.
5. Write per-seed summaries, hashes, and failure reasons.

G5 requires a valid matched comparison and at least 15 successful paired seeds;
it does not permit substituting new seeds because the results are inconvenient.
Aggregate all 20 planned seeds and explicitly report missing or invalid ones.

Exit: G4 and G5 pass and every machine aggregate links to per-seed source
artifacts.

## 9. Phase 4 - integrated inference, figures, and audit

Estimated elapsed compute: 2-4 days. Interpretation and manuscript work follow
without keeping the GPU instance continuously rented if all artifacts are
backed up.

1. Regenerate all human and machine aggregate tables from sealed partitions.
2. Run primary statistics, cluster/permutation analyses, directional Bayes
   factors, incremental prediction, and architecture/intervention comparisons.
3. Generate main and supplementary figures only from immutable aggregate
   tables.
4. Run a clean reproduction of critical targets in a new output directory.
5. Compare hashes between the original and clean reproduction.
6. Produce a deviation log separating planned, necessary engineering, and
   exploratory analyses.
7. Create a final provenance inventory containing code/config/environment
   hashes, source manifests, participant flow, exclusions, failures, sealed
   metrics, statistics, figure data, and logs.
8. Verify off-host copies by rehashing them before releasing or deleting the
   AutoDL instance.

Completion means `results/study_complete.flag` exists, G6 passes, and at least
two independently located copies of irreplaceable outputs have matching
checksums.

## 10. Calendar and rental recommendation

| AutoDL days | Planned work |
|---|---|
| Day 0-1 | Phase 0 hardening, freeze, backup test |
| Day 1-5 | pilot downloads, source contracts, three participant/five-fold pilots |
| Day 4-10 | full downloads, selective extraction, early human preprocessing |
| Day 6-23 | complete human preprocessing and five-fold analyses |
| Day 6-15 | machine benchmarks and 80-run queue when GPU is otherwise idle |
| Day 20-27 | aggregation, H1-H5, machine comparison, figures |
| Day 27-35 | retries, clean reproduction, backup verification, contingency |

Expected continuous AutoDL time from this point is **21-35 days**. Rent or
retain 30 days initially if pricing permits, with a one-week contingency. Do
not prepay beyond that solely on this estimate: revise it after the three
human pilots and four production-scale machine benchmarks.

## 11. Monitoring cadence and pause conditions

The unattended queue writes a concise heartbeat at least every 10 minutes:
active target, participant/seed/fold, elapsed time, RSS, VRAM, free disk, latest
checkpoint, and retry count. Review a daily summary rather than relying on an
open SSH or browser session.

Pause automatically when any of these occurs:

- free scratch below 500 GiB;
- RSS above 85 GiB or an OOM event;
- GPU temperature/power errors or repeated CUDA faults;
- three failures of the same target;
- checksum mismatch or changed source archive;
- inconsistent event/trial reconciliation;
- leakage, duplication, or configuration-drift audit failure;
- output growth exceeds the pilot projection by 25%.

## 12. Immediate next actions

1. Patch the environment audit to use cgroup limits and add workflow resource
   declarations.
2. Freeze the current remote source/config/environment snapshot.
3. Generate complete acquisition manifests, but download only the three pilot
   cases first.
4. Inspect real events/channels and fill the gated YAML fields and participant
   table.
5. Run one participant and five folds per dataset with resource telemetry.
6. Review G1/G2 and replace provisional disk/time estimates with measurements.
7. Only then launch full human acquisition and the benchmarked 80-run machine
   queue.

