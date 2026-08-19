# Unified Science Advances figure plan

## Editorial premise

This is one study with two linked analytical stages, not an original study
followed by a visually detached extension. The characterization stage identifies
the human access geometry and ranks five independently trained computational
organizations. The adaptation and causal-testing stage then asks whether each
organization can acquire that geometry
through an outcome-isolated human-neural target while retaining task function
and causal subspace specificity.

The main text uses six figures. Reducing eight sequential figures to six removes
repeated architecture profiles, repeated sample summaries, and a false visual
break between the analytical stages. Full QC, distributions, exhaustive seed views,
and secondary inferential summaries remain available in the supplement.

## Main figure sequence

### Figure 1 - One study, linked analytical stages

- Human and machine arms converge only at the common four-metric geometry.
- Characterization ranks the baseline models; adaptation and intervention test mechanism.
- The five falsifiable architecture patterns and the discovery-to-adaptation
  evidence ladder are visible on the same page.
- Human data do not train the baseline models; held-out geometry never enters
  adaptation-stage checkpoint selection.

### Figure 2 - Human access geometry

- Discovery and replication time courses appear together.
- Participant-level primary-window effects replace isolated significance labels.
- The four-metric fingerprint, leave-one-dataset-out distance, and directional
  agreement make heterogeneous replication explicit rather than silently pooled.
- Incremental prediction is shown alongside the geometry result so a positive
  condition contrast is not mistaken for useful trial-level prediction.

### Figure 3 - Controlled machine geometry

- Five architecture profiles show the competing computational claims.
- Accuracy matching and 20 independent paired seeds establish comparability.
- Targeted-versus-random intervention establishes causal relevance.
- The capacity-limited and unlimited shared-state conditions remain visibly
  separate, preserving the central capacity-control falsifier.

### Figure 4 - Baseline theory competition

- Human and machine metric profiles are overlaid once, not repeated elsewhere.
- Seed-level RMS distance and cosine concordance jointly rank architectures.
- The constrained-versus-unlimited capacity test retains its explicit
  equivalence-based falsifier.

### Figure 5 - Human-guided adaptation

- The anti-circular train/adapt/evaluate flow leads the figure.
- Task-trained, human-adapted, and matched-sham stages appear together.
- Human-specific gain, complete-case sensitivity, and the retention gate test
  target specificity rather than generic optimization.

### Figure 6 - Study-level synthesis

- Adaptation efficiency and task retention establish functional cost.
- Pre/post causal intervention tests mechanistic preservation.
- Cross-dataset transfer tests whether acquisition generalizes beyond discovery.
- The closing evidence map distinguishes baseline resemblance, acquisition, and
  causal preservation, including explicit null or falsifying outcomes.

## Supplementary sequence

- Figure S1: outcome-blind preprocessing and retention QC.
- Figure S2: complete human metric and Access Index distributions.
- Figure S3: complete machine seed sensitivity, accuracy bins, and intervention
  distributions.
- Figure S4: prediction, permutation evidence, Bayes factors, and sample
  composition formerly occupying the old main Figure 6.
- Figure S5: adaptation convergence, optimization diagnostics, sham balance,
  and complete architecture-by-seed trajectories.
- Figure S6: robustness to normalization, distance definition, dimensionality,
  temporal window, and dataset rotation.

## Visual grammar

- Architecture colors are immutable throughout all main and supplementary
  figures; dataset identity is conveyed by labels, position, and limited
  secondary accents.
- Every main figure has one declarative headline and one short evidential
  subtitle. Panel headings state the claim, not merely the chart type.
- Panel letters are lowercase, bold, aligned to a shared top-left anchor.
- A pure white canvas, open axes, direct annotation, seed/participant points,
  and interval estimates replace default plotting styles.
- Sand is reserved for sham observations and the outlines of equivalence gates
  or control regions; it is not used as a page or panel background.
- No decorative gradients, 3-D effects, radar plots, or redundant legends.
- Vector PDF and SVG plus 320-dpi PNG are generated from the same R source.

## Production status

The native-R production set is complete. A Python exporter freezes 22
renderer-neutral source tables; `ggplot2` and `patchwork` render six main figures
to PNG, PDF, and SVG. The renderer uses only `#C22525`, `#3F3A39`, `#6F5E56`,
`#C3AB8C`, and `#E1D6C7` on a pure white canvas. A machine-readable manifest
records package versions, file sizes, and SHA-256 hashes. The older Python
mockbook is retained only as a historical fallback and is not the publication
figure source.
