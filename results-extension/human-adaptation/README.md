# Human-guided adaptation stage

This is a linked analytical stage of one study, not a separately labelled
experiment. The baseline characterization under `results/` supplies frozen
machine lineages and the human geometry target; those files are hashed before
adaptation and verified again at completion.

The adaptation target is a lower-level temporal representational-similarity distribution
computed directly from preprocessed Gabor EEG activity. It is not any of the four final
Jacobian geometry metrics, the scalar Access Index, or a human–machine geometry distance.
Participant-held-out final geometry is evaluation-only.

AutoDL progress is recorded in `queue-state.json`. Final machine-readable tables, tests,
figures, and article digests appear under `aggregate/`, `statistics/`, `figures/`, and
`article/` when the queue completes.

The publication figure path is `figures/science-advances-r/final/`. The queue
first exports renderer-neutral CSV tables and then renders six figures natively
with R. Retention-gate failures remain in the all-runs analysis and are audited
again in gate-compliant and complete-case sensitivity analyses.
