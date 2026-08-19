# Implementation roadmap

## Objective

Produce a reproducible ordinary study that can be developed and reviewed
locally, uploaded as a small code archive, and executed on one AutoDL RTX 5090
host. The laptop is the control and development machine; AutoDL is the
production compute and data host.

## Completion update (2026-08-17)

The ordinary-study production workflow is complete. Phases 1-5 passed with 173
human participants (865/865 folds), 100/100 machine runs, five primary
frequentist and Bayesian analyses, cross-dataset replication, nested prediction,
theory comparison, publication figures, and frozen provenance. The phase text
below is retained as the historical implementation plan rather than a current
status report.

## Human-neural adaptation and causal-testing stage (launched 2026-08-16)

Status: **production adaptation running on AutoDL**

- Phase E0: audit the baseline characterization, hash frozen outputs, and construct participant
  splits — complete.
- Phase E1: implement lower-level human target, task retention, parameter
  anchoring, matched sham, exact random reconstruction, and tests — complete.
- Phase E2: one-architecture/one-seed/one-fold smoke test — passed.
- Phase E3: freeze the extension configuration and source provenance — complete.
- Phase E4: 5 architectures × 20 seeds × 5 participant folds × 2 adaptation
  conditions — running and resumable.
- Phase E5: random/task/human/sham geometry, external transfer, and causal
  intervention analyses — queued.
- Phase E6: paired seed-level statistics, figures, result digest, immutability
  verification, and SSD archive — queued.

The extension is an ordinary research analysis, not a Registered Report. Its
adaptation loss and checkpoint selection are prohibited from using gain,
broadcast, persistence, concentration, Access Index, or final human-machine
distance.

## Phase 0 - specification correction and local production core

Status: **source complete; AutoDL PyTorch verification pending**

- Reframe the project as an ordinary study.
- Preserve primary hypotheses as prespecified analyses, not Registered Report
  commitments.
- Correct the synthetic pilot so it uses the actual registered formulas.
- Fix machine output-head count and internal-state shape contracts.
- Implement analytic Jacobians, ordered propagation, geometry metrics and
  rotation-invariance tests.
- Create an AutoDL bootstrap and environment-verification command.
- Implement human rollout training, restart checkpoints and held-out QC.
- Implement condition-blind inputs, output readouts and chunked Parquet metrics.
- Implement the procedural task and four parameter-matched machine systems.
- Exercise fold/PCA/readout/metric sealing in a synthetic end-to-end fixture.

Exit criteria:

- local core checks pass;
- no target data are required;
- metric formulas have independent unit tests;
- the repository can be archived without data or generated results.

Achieved locally on 2026-07-30: 28 tests passed and three PyTorch-only tests
were skipped because the local environment has no PyTorch. The skipped tests
are mandatory in the AutoDL `stage2_ready` target.

## Phase 1 - dataset contracts and one-participant dry runs

Status: **all generic code drafted; real-file verification is the next action**

Estimated effort: 4-8 working days.

- Create immutable manifests for the three target repositories. **Implemented.**
- Implement verification-gated repository adapters and source inspection.
  **Implemented; source-specific mappings remain intentionally blank.**
- Normalize participant, trial, event and channel identifiers.
- Add MNE readers and outcome-blind preprocessing. **Implemented.**
- Run one participant from each dataset through preprocessing and inspect QC.
- Record actual peak RAM, scratch use and elapsed time.

Exit criteria:

- three one-participant smoke tests pass;
- events reconcile with source counts;
- no awareness/report field reaches preprocessing or primary model inputs;
- resource measurements update the production schedule.

## Phase 2 - human analysis

Status: **source complete; target-data execution pending**

Estimated compute: 10-18 days on one RTX 5090/25+ CPU host, including retries.

- Preprocess and cache all eligible participants.
- Assign deterministic five-fold splits.
- Fit training-only PCA/whitening and residual dynamics.
- Verify analytic Jacobians against autograd.
- Stream Jacobian propagators in chunks; do not persist full dense tensors.
- Seal trial-time metric Parquet files and join condition labels.
- Run prespecified H1-H5 analyses and outcome-neutral quality controls.

Exit criteria:

- every output has configuration, input and code hashes;
- all exclusions and model-QC failures are reported;
- primary statistics can be regenerated from sealed metric tables alone.

## Phase 3 - machine experiment

Status: **source complete; CUDA smoke and production execution pending**

Estimated compute: 4-8 days, partly parallel with Phase 2.

- Generate deterministic image batches on demand.
- Train four parameter-matched architectures across 20 paired seeds.
- Match performance using the prespecified threshold-bin rule.
- Compute exact bottleneck-state Jacobians at six steps.
- Compare machine and human signatures.
- Run top-four versus random-subspace interventions.

Exit criteria:

- architecture parameter counts are within 10%;
- all 80 training runs are checkpointed and resumable;
- accuracy matching and intervention success counts are reported per seed.

## Phase 4 - inference, audit and publication package

Status: **analysis and figure scripts drafted; manuscript/results pending**

Estimated effort: 2-4 weeks, overlapping the final compute week.

- Run directional Bayes factors and permutation analyses.
- Produce all primary and supplementary figures from immutable result tables.
- Run a clean reproduction of critical targets.
- Write limitations, deviations and null outcomes without result-dependent
  pipeline changes.
- Archive source, lockfiles, manifests, metrics allowed by source licences and
  figure data.

## Realistic calendar

- Local preparation before AutoDL: 2-4 weeks.
- First AutoDL data integration and complete computation: 3-5 weeks.
- Audit, interpretation and manuscript: 2-4 weeks.
- Total active project time: approximately 8-12 weeks, excluding external
  review, collaborator delays and repository outages.

## Stop/go gates

1. Do not launch the full dataset until three participant smoke tests pass.
2. Do not launch 80 machine runs until two seeds per architecture pass.
3. Do not open primary contrasts until model-QC and leakage reports are frozen.
4. Do not delete raw archives or irreplaceable outputs from AutoDL local disk
   until verified copies exist in reliable external storage.
