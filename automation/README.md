# AutoDL full-study queue

The queue is a single resumable supervisor for Phases 0-4. It owns one active
scientific/compute command at a time and uses Snakemake checkpoints underneath.

## Commands

Start or resume:

```bash
cd /root/autodl-tmp/jacobian-conscious-access
bash automation/start_autodl_queue.sh
```

Start or resume with the self-restarting day watchdog:

```bash
bash automation/start_day_queue.sh
```

Inspect current state:

```bash
bash automation/status_autodl_queue.sh
tail -100 logs/full-study/supervisor.log
```

Stop gracefully at the current command boundary:

```bash
bash automation/stop_autodl_queue.sh
```

Restarting is safe. Completed command receipts with `status: PASS` are reused;
downloads resume from `.part` files, Snakemake uses `--rerun-incomplete`, and
model training resumes from atomic checkpoints.

## State contract

The authoritative state is under `automation/state/full-study/`:

- `queue.pid`: actual supervisor process ID;
- `queue.status.json`: current phase/command, resource heartbeat, and outcome;
- `phase0.status.json` through `phase4.status.json`: phase gates;
- `commands/*.json`: command, attempt, exit code, peak RAM, minimum free disk,
  elapsed time, and log path.

The queue samples cgroup RAM, free scratch, and GPU telemetry every 15 seconds.
It terminates the current process group above 85 GiB RAM or below 500 GiB free
scratch. It never deletes data to recover space.

## Scientific gates

Phase 1 downloads and inspects bounded public pilots. If mappings, channel
groups, contrasts, outcomes, or participants are unverified, the queue writes
`WAITING_REVIEW` and continues Phase 3 so the GPU is not idle. It polls the
adapter gate every 60 seconds after Phase 3.

To unlock Phase 2, update and review:

- `configs/datasets/gabor.yaml`;
- `configs/datasets/somato.yaml`;
- `configs/datasets/kronemer.yaml`;
- `configs/execution/participants.tsv`.

Do not set `adapter_status: verified` until the source-inspection evidence
supports every mapping. Once `scripts/check_adapter_gate.py` passes, no queue
restart is needed.

Phase 2 runs one included participant per dataset through all five folds before
full acquisition. Any pilot at or above 80 GiB cgroup RAM blocks expansion.

Phase 3 first trains one production-scale seed per architecture, verifies the
parameter tolerance, and then queues all four architectures by 20 paired seeds.
Scientific failures are retained and do not receive substitute seeds.

Phase 4 builds the final figures and study-completion target, audits expected
human and machine outputs, and freezes the final source/configuration provenance.

For the linked adaptation stage, finalization also runs the retention-gate
sensitivity audit, exports 22 renderer-neutral figure tables, and invokes
`scripts/render_science_advances_figures.R`. The native-R six-figure set is the
primary publication output; the older Python integrated renderer is no longer
called by the automatic queue.

Each trained seed writes validation presence accuracy by difficulty bin. The
production audit selects, among bins whose four architecture means differ by no
more than 0.02, the common bin closest to 0.70 validation accuracy. If no bin
qualifies, it records the prespecified all-bin fallback, stabilized
inverse-probability weights, and test-accuracy covariates.

HTTP manifests use eight resumable byte-range workers when the source advertises
range support. A pre-existing sequential `.part` file becomes an immutable
prefix; only its remaining ranges are fetched. Final files are assembled
atomically and receive `.download.json` receipts containing remote metadata and
SHA-256. Temporary range parts are removed only after the final receipt passes.
