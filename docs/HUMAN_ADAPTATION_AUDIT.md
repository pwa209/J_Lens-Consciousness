# Human-neural adaptation implementation audit

Date: 2026-08-16

## Frozen baseline-characterization inputs

The completed baseline characterization contains 173 included participants (865 human folds),
100 trained machine lineages, the original machine interventions, theory
comparison, statistics, and figures. The adaptation stage writes only below
`results-extension/human-adaptation/`. Before the extension smoke test, every
file below `results/` was recorded by byte size and SHA-256. The queue verifies
that manifest again before declaring the adaptation stage complete.

## Data and checkpoint audit

- All five architectures have 20 task-trained `model.pt` checkpoints.
- Exact random initialization is reconstructable because the baseline stage seeded
  Python, NumPy, CPU Torch, and CUDA Torch immediately before architecture
  construction. Determinism is tested directly.
- The governed Gabor derivatives retain preprocessed trial × sensor × time EEG
  arrays at 100 Hz. The extension roster has 28 included Gabor participants.
- Baseline fold assets retain training-only PCA parameters but not latent trial
  streams. The adaptation stage therefore derives its target from the governed
  preprocessed sensor arrays without modifying or regenerating the baseline
  characterization.

## Alignment decision

Stimulus-level EEG–machine pairing is not defensible: the procedural machine
task and the human Gabor experiment are related detection paradigms but do not
share item identities. A distributional lower-level target is therefore used.

For each participant, six EEG samples from 150–300 ms are selected. Each
trial's sensor pattern is spatially centered and L2-normalized, then its 6 × 6
time-by-time cosine matrix is calculated. Participant-equal pooled means and
variances of this matrix form the adaptation target. This target:

- is dimension-independent and needs no arbitrary EEG-to-machine axis mapping;
- uses direct neural activity, not outcome contrasts or final geometry;
- is computed separately for training and inner-validation participants;
- never loads outer held-out participants during adaptation;
- contains none of gain, broadcast, persistence, concentration, Access Index,
  RMS human distance, final cosine similarity, or magnitude ratio.

The matched sham jointly permutes the six time labels while preserving matrix
values, optimization batches, step budget, task loss, and parameter anchor.

## Participant cross-fitting

The 28 Gabor participants are deterministically balanced over five outer folds
(6, 6, 6, 5, and 5 held out). Each non-held-out set contains five inner
validation participants and 17 or 18 adaptation participants. Sets are
disjoint, collectively exhaustive, and covered by a unit test. Public split
artifacts retain only counts and truncated participant hashes.

## Trainable scope and retention

The common encoder, task-cue projection, and all task readout heads are frozen.
Architecture-specific state-formation and transition parameters are trainable.
AdamW uses learning rate 1e-5, weight decay 1e-4, batch size 1024, at most 150
steps, task-loss weight 1.0, and parameter-anchor weight 0.001. Checkpoint
selection uses only inner-participant neural alignment loss. Absolute presence
accuracy change at the frozen difficulty bin 2 must not exceed 0.02; failures
are retained and reported rather than silently discarded.

## Pre-analysis amendment

After production adaptation began but before any outer-held-out geometry was
computed, `stage_comparison.py` was corrected to populate the required
`task_accuracy` field from each stage's frozen test-performance table. This was
an output-schema correction and did not alter targets, splits, checkpoints,
training, geometry definitions, hypotheses, or statistical endpoints.

## Final retention and sensitivity audit

Production completed 1,000 outer-fold runs. Thirty-four runs were outside the
absolute ±0.02 task-accuracy retention gate: 17 human-target and 17 matched-sham
runs. They were not silently removed. The final report carries three views:

1. all runs, preserving the intention-to-adapt comparison;
2. gate-compliant folds, requiring at least three retained outer folds per seed;
3. complete-case seeds, retaining only lineages with all five human-target and
   all five sham folds inside the gate.

No architecture had a positive Holm-corrected human-adaptation or
human-versus-sham effect in any view. The capacity-limited shared-state sham
contrast was nominal in the all-runs view (+0.001242, p = 0.0281) but not after
Holm correction (p = 0.1405); it weakened under the gate-compliant view
(+0.000226, p = 0.389) and complete-case view (+0.000876, p = 0.0587; Holm
p = 0.293). The article therefore treats it as non-robust and does not use it
as evidence for human-specific acquisition.
