# Implementation decisions

## 2026-07-30 - ordinary-study status

The implementation does not claim Registered Report status. Target-data access
is not conditional on acceptance in principle. The original condition-blind
model-fitting and label-separation rules are retained because they reduce
circularity and leakage.

## 2026-07-30 - synthetic pilot is not evidential

The packaged pilot is treated as a historical proof of concept, not as a
validated implementation of the primary metrics. The production repository
uses independent tests for every formula and will regenerate a corrected
synthetic result.

## 2026-07-30 - streaming Jacobians

Full `[trial, time, horizon, 32, 32]` propagator tensors are transient. They are
computed in chunks and reduced to metric tables. Only small diagnostic samples
may be retained. This prevents avoidable disk and memory growth without
changing any statistic.

## 2026-07-30 - five machine heads

The machine task has five heads: presence, orientation, location, contrast bin
and delayed action. The earlier schematic phrase "4 nonverbal outputs" is
superseded.

## 2026-07-30 - machine state contract

Exact machine Jacobians are taken with respect to an explicit 32-dimensional
integration bottleneck at each processing step. Architecture-specific high-
dimensional convolutional features are not used as the differentiated state.
This makes memory bounded and the cross-architecture comparison well defined.

## 2026-07-30 - delayed action operationalization

The fifth machine head predicts target presence at the final processing step
and is not scored at earlier steps. The task cue is a five-way one-hot vector;
the cued head receives additional loss weight while all heads remain present.
This is the simplest deterministic operationalization of the underspecified
"delayed action" phrase in the source protocol and must be reported explicitly.

## 2026-08-14 - bounded-memory continuous EEG preprocessing

Continuous recordings are anti-alias resampled to the planned 100 Hz analysis
rate before the 0.5--40 Hz FIR filter, PyPREP, and ICA steps. The original order
caused repeatable SIGKILL failures while filtering a 257-channel, 6,651-second
Kronemer recording at 1000 Hz because MNE allocated several full-rate work
arrays. The revised order reduces subsequent array sizes by approximately ten
fold; the anti-aliasing resampler precedes decimation and the requested final
analysis bandwidth is unchanged. This outcome-blind engineering decision was
made from process failure and memory evidence without inspecting condition
contrasts. Each affected participant records the ordering in its QC deviations.

## 2026-08-14 - fully missing Kronemer reference channel

The final Kronemer sensor channel (index 256) is entirely non-finite in both
pilot files and is not present in any verified output channel group. Before
training-fold PCA, an all-non-finite channel is deterministically replaced with
zero so it contributes no variance while source channel indices remain stable.
Partially non-finite channels cause an explicit failure rather than imputation.
This was identified from numerical validation, before any condition contrast
was inspected, and is recorded in each fold summary.

## 2026-08-14 - line-noise control before early resampling

The memory-bounded early resampling path now applies a channel-wise zero-phase
60-Hz IIR notch to continuous Kronemer data before downsampling from 1000 Hz to
100 Hz. This prevents line noise from aliasing toward the 40-Hz analysis edge
without recreating the whole-recording FIR memory spike. The 0.5--40 Hz
analysis band, 100-Hz final sampling rate, and all condition-blind processing
rules are unchanged. The applied ordering is recorded in preprocessing QC.

## 2026-08-14 - Kronemer run-level source mapping

The full outcome-blind filename audit found 19 unresolved EEG-to-behavior
links among 349 Kronemer recordings. The irregularities were limited to source
organization: spaces versus underscores, four conditions whose second CSV was
mislabeled Session 1, early report-task recordings stored flat while behavior
was grouped by run, and one recording with no behavioral CSV. Mapping now uses
exact directory or normalized session evidence first, then within-condition
acquisition order only when EEG and behavioral file counts agree. Combined-run
EEG files may map to multiple behavioral CSVs. Calibration recordings and the
single behavior-less recording are excluded at run level. Every included and
excluded source link, including its mapping method, is written to the
participant descriptor. No behavioral outcomes or condition contrasts were
used to define these rules.

## 2026-08-14 - Somato channel-location filename variant

Somato participant 11 uses `EEG_selezionato_locations.mat` for the same
per-epoch channel-location structure named `EEG_locations.mat` elsewhere.
The adapter now accepts either non-resource-fork `*locations.mat` variant,
prefers the canonical name when present, and records the selected path in the
participant descriptor. This is a source-format compatibility correction and
does not alter signal values or condition labels.

## 2026-08-14 - source-completeness roster exclusions

The full structural audit found two Gabor participants with incomplete
BrainVision file sets (`sub-37`, `sub-41`) and three extracted Kronemer
participants with no task EEG recording (`579_NRP`, `587_NRP`, `610_NRP`).
The production-roster builder now marks these five participants `include=0`
with an explicit source-exclusion reason. This reduces the analyzable roster
from 181 to 176 before preprocessing QC and without reference to behavioral
outcomes or condition effects.

## 2026-08-14 - Kronemer clock-alignment run exclusion

The 10-ms maximum affine clock-residual threshold remains unchanged. When an
individual Kronemer EEG/behavior pair exceeds it, that pair is now excluded at
run level and its exact error is recorded in the participant descriptor;
other independently aligned runs for the participant remain eligible. This
prevents one malformed run from aborting the full participant or the study
while preserving the prespecified timing-quality boundary.

## 2026-08-14 - Gabor interrupted trigger window

Gabor sub-31 trial 156 contains the target and early stimuli followed by two
`New Segment` acquisition markers, with no block or awareness outcome trigger.
The adapter now excludes only target-trigger windows containing an explicit
recording discontinuity, records their trial index and reason in source
provenance, and continues to fail on any other missing block/outcome pattern.
This source-quality exclusion was defined from trigger structure alone; no
condition result was inspected.
