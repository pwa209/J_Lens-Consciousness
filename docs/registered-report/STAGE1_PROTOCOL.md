# Stage 1 protocol: competing geometries of conscious access

## Status and prospective boundary

This is a candidate Level 2 Registered Report, subject to editorial eligibility
assessment by PCI Registered Reports or Cortex. The prospective boundary is
2026-08-11. Eighty runs from four machine architectures were complete before
that boundary. A fifth architecture was designed after the boundary and before
any confirmatory human group analysis. Gabor participant sub-10 completed five
pipeline folds as a methods pilot; preprocessing QC and fold-3 model QC were
inspected. Somato sub1 processing failed before completion. No human group
contrast or human-machine ranking may be inspected before in-principle
acceptance (IPA). All four named pilot participants are excluded from the
confirmatory samples.

The machine results are prior work, not prospectively registered outcomes. The
prospective claims concern held-out human participants, replication across
datasets, and human-machine comparisons whose human side is unavailable at
Stage 1. The editor must decide the applicable bias-control level.

## Central question

Which computational organization—feedforward processing, recurrent processing,
local private modules, a capacity-limited shared workspace, or an
unlimited-capacity shared state—best predicts the geometry of conscious access
in held-out human EEG, and does the answer replicate across report regime and
sensory modality?

Architectures are model classes inspired by theoretical claims, not exhaustive
or unique implementations of entire consciousness theories.

## Competing model classes and falsifiable predictions

The four primary metrics are gain, broadcast, persistence, and rank-4
concentration. Effective rank is secondary and is reported as a
dimension-normalized quantity. Predictions concern accessible-minus-inaccessible
contrasts in the prespecified late window unless a dataset-specific positive
control window is named.

| Model class | Claim represented | Gain | Broadcast | Persistence | Concentration | Result that favors it |
|---|---|---:|---:|---:|---:|---|
| Feedforward | access can be explained without recurrent integration | + | 0/+ | 0 | 0/+ | human effects are transient and it has the smallest frozen-vector distance |
| Recurrent | recurrent processing is sufficient | + | + | ++ | + | recurrence matches humans without global workspace topology |
| Private modules | local sufficiency without inter-module sharing | module-local + | 0 | + | ++ | human geometry is concentrated and non-broadcast |
| Constrained shared workspace | limited shared capacity produces global access | ++ | ++ | ++ | ++ | uniquely closest human match, including higher concentration than unlimited sharing |
| Unlimited shared state | sharing, not a bottleneck, is sufficient | ++ | ++ | ++ | 0/+ | equals or beats the constrained workspace despite higher capacity |

Symbols are ordinal directional predictions, not assumed effect sizes. A
capacity-limited workspace interpretation is rejected if the constrained and
unlimited shared-state models are equivalent on all four primary standardized
human-machine distances within ±0.20 discovery-human SD, or if the unlimited
model is closer. A unique workspace interpretation is also rejected if the
recurrent model is no farther from humans within the same equivalence margin.
Private-module or feedforward superiority favors the corresponding alternative
and is reported as such.

## Architecture control

The new `unlimited_shared_state` model retains four specialist states, shared
task and stimulus generation, six processing steps, output heads, optimizer,
approximately two million trainable parameters, and the specialist/shared update
topology. Its shared workspace is 128-dimensional—the combined capacity of four
32-dimensional specialists—rather than the constrained model's 32 dimensions.
Twenty seeds are run. Accuracy matching and causal-subspace checks are repeated
over all five architectures.

## Discovery and replication

Gabor is the discovery dataset because it provides the most direct visual
seen/unseen contrast. Gabor sub-10 is excluded. All metric transformations,
signs, time summaries, equivalence bounds, and human-machine scaling constants
are fitted or instantiated using training folds and eligible Gabor discovery
participants only, then frozen.

Kronemer is replication 1, testing generalization across report/no-report visual
paradigms. Somato is replication 2, testing out-of-modality generalization.
Neither replication dataset may modify the frozen signature. Each replication
is tested separately. A secondary leave-one-dataset-out analysis trains the
fixed equal-weight signature on two datasets and evaluates it unchanged on the
third; it cannot replace a failed primary replication.

## Comparability and normalization

PCA, whitening, residual dynamics, output maps, and baseline normalization are
fit on each participant's training fold only. Raw primary metrics are transformed
as follows: log gain; logit broadcast; Fisher transform of the square root of
persistence; and logit concentration. Effective rank is divided by state
dimension and logged. Values are centered and scaled by that participant's
training-fold pre-stimulus baseline. A scale floor of 0.0001 is fixed.

Held-out folds receive only their corresponding training-fold transforms.
Fold-level estimates are averaged with equal weight within participant, and
participants receive equal group weight. Rotation is immaterial by construction;
the transformations address scale, and dimension-normalized effective rank
addresses differing latent capacity.

For human-machine comparison, the primary vector contains the four transformed
condition contrasts with equal component weights. Component scaling is the SD
across eligible Gabor discovery participants and is frozen before replication;
no architecture contributes to this scaling. Primary similarity is standardized
Euclidean distance. Cosine similarity after L2 normalization is a secondary
shape-only result, and the Euclidean norm ratio separately reports magnitude.
Thus shape and effect magnitude cannot compensate for each other silently.

## Pilot firewall

Before IPA, permitted pilot information is limited to preprocessing QC,
exclusion feasibility, missingness, runtime, RAM/GPU/disk use, optimizer
convergence, epochs, validation loss, and failures. Forbidden information is any
condition-specific geometry, contrast, effect size, group result, architecture
ranking, or human-machine similarity. The content-blind inventory created by
`automation/seal_human_pilot.py` hashes every existing pilot file without parsing
its scientific content. The AutoDL human queue requires a public IPA receipt and
cannot resume merely because a Stage 1 manuscript was submitted.

## Confirmatory decision sequence

1. Apply the frozen analysis to eligible Gabor participants and estimate the
   four-metric discovery signature.
2. Freeze the Gabor scaling constants and compare all five machine classes using
   accuracy-matched trials; report equivalence tests and complete rankings.
3. Apply the unchanged signature to Kronemer and Somato independently.
4. Claim cross-dataset replication only when both replication estimates have the
   predicted sign and their uncertainty excludes the registered null/equivalence
   boundary as specified in the Stage 1 statistical appendix.
5. Report LODO generalization, cosine similarity, magnitude, effective rank,
   alternative preprocessing, and multiverse analyses as secondary.

## Stage 1 deliverables before human resumption

- editor-confirmed eligibility and bias-control level;
- manuscript with the disclosure above and machine results clearly labeled prior;
- immutable code/config archive and pilot hash inventory;
- sampling/exclusion rules and power/sensitivity analysis for all three datasets;
- complete statistical appendix, including multiplicity and missing-data rules;
- public Stage 1 IPA URL recorded by `automation/release_stage1.py`.

