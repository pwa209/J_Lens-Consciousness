# Ordinary research article: Phase 1-5 execution plan

## Scientific framing

The article is a confirmatory, ordinary research report comparing five model
classes: feedforward, recurrent, private modules, a capacity-limited shared
workspace, and an unlimited shared state. The completed machine work and pilot
processing are reported transparently as development work, but no claim of
preregistration or Registered Report status is made.

The central question is whether human conscious-access geometry uniquely
requires a limited-capacity shared workspace, or whether recurrence, local
processing, or unrestricted sharing explains the human pattern as well or
better. Gabor is the discovery dataset, Kronemer is visual report/no-report
replication, and Somato is cross-modal replication. Pilot participants remain
excluded to keep development and final estimation separate.

## Phase 1: complete acquisition and roster

- Resume-safe download of the full Gabor and Kronemer sources.
- Reuse and inventory the full Somato preprocessed archive.
- Selectively extract required Kronemer files.
- Build the acquired participant roster while excluding the four development
  pilots.
- Hash the raw-data manifests and enforce a 500-GiB free-space floor.

## Phase 2: five-architecture machine completion

- Allow the already-running 20-seed unlimited shared-state training to finish.
- Recompute persistence for all 100 machine checkpoints.
- Rebuild accuracy matching and causal-intervention summaries.
- Treat a failed theoretical/intervention prediction as a result, not as an
  execution error; only missing or corrupt outputs stop completion.

## Phase 3: pipeline validation

- Run all five folds for the four excluded development participants.
- Verify the corrected epoched-MNE filtering path on Somato.
- Verify both report and no-report Kronemer paths.
- These results never enter confirmatory aggregation.

## Phase 4: full human and theory analysis

- Standardize and preprocess every acquired participant outcome-blind.
- Freeze the final roster using preprocessing QC before condition contrasts.
- Run five-fold within-participant cross-fitting and held-out Jacobians.
- Aggregate only participants in the QC-frozen roster.
- Run primary frequentist and Bayesian contrasts, nested prediction, the
  discovery/replication analysis, and the five-theory human-machine comparison.
- Separate execution completeness from whether a scientific prediction passed.

## Phase 5: article products and audit

- Generate main and supplementary figures.
- Generate the five-theory ranking, capacity-equivalence decision, replication
  summaries, and a manuscript-ready factual results digest.
- Run final human and machine inventories.
- Freeze provenance and write `results/study_complete.flag`.

## Continuous execution

`automation/run_ordinary_study.sh` starts a single resumable supervisor. Every
phase writes atomic status and command receipts. Transient failures retry with
exponential backoff. Completed Snakemake outputs are retained, downloads resume,
and the queue advances automatically through Phase 5. GPU-heavy human work is
held until machine work finishes; full acquisition can proceed concurrently
with the active machine queue.

