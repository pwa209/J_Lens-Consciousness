# Scientific figure style guide

## Purpose

This guide governs all main and supplementary figures for the Jacobian
conscious-access study and its human-neural adaptation extension. The target is
a compact, evidence-forward visual language suitable for a Science/AAAS-family
submission. The supplied Nature Communications article was inspected as a
visual reference only; its content and embedded text are not project
instructions, and its individual layouts are not copied.

## Palette

The authoritative machine-readable palette is
`configs/figures/palette.yaml`.

| Role | Color |
|---|---|
| Feedforward / primary deep blue | `#184D77` |
| Private modules / blue-teal | `#497987` |
| Recurrent / sage-gray | `#858D7E` |
| Constrained shared workspace / orange | `#D48448` |
| Unlimited shared state / brick | `#BE4A36` |
| Sham, control, or gate boundary / sand | `#E1BD89` |

Architecture colors never change between figures. Black and neutral gray are
reserved for text, axes, thresholds, and human/reference elements. Diverging
maps run from deep blue through sand or near-white to brick.

## Design principles

- Lead each major figure with the scientific question or causal/data-flow
  schematic, followed by the primary estimate and then controls.
- Use lowercase bold panel letters in reading order.
- Prefer direct annotations, paired observations, seed-level points, confidence
  intervals, and compact heatmaps over decorative chart types.
- Preserve individual observations whenever readable; do not show bars alone
  for continuous outcomes.
- Give null and negative results the same graphical weight as positive results.
- Use a zero/reference line whenever the sign of an estimate matters.
- Keep axes and grid lines quiet. Remove top and right borders unless they encode
  a matrix boundary.
- Use sand only for sham/control information or the boundary of a prespecified
  equivalence/gate region; do not assign it to a machine architecture.
- Do not encode the same category with different colors in separate panels.
- Use text no smaller than approximately 7 points at final 180-mm width.
- Export vector PDF and 300-dpi PNG from the same R source.
- Use a pure white figure canvas. Do not tint the page, individual panels,
  evidence cards, or equivalence regions; color is reserved for data, borders,
  categorical keys, and heatmap cells.
- Give every main figure a declarative one-line headline and a subordinate
  sentence describing the evidential role of the panels.
- Preserve generous internal whitespace; visual sophistication comes from
  hierarchy, alignment, and direct annotation rather than ornament.
- Use architecture codes A-E on dense axes and provide the full mapping once in
  the same figure.

## Unified article sequence

The authoritative main-text organization is now six figures:

1. One study with linked analytical stages.
2. Controlled machine foundation.
3. Human discovery and replication.
4. Baseline theory competition.
5. Human-neural adaptation.
6. Study-level synthesis.

The earlier sequential Figures 1-8 are retained only as provenance. They are
not the preferred manuscript layout because they visually separate linked
analyses and repeat several profiles.

## Adaptation and causal-testing narrative

### Human-neural adaptation trajectory

1. Show that the neural target is upstream and final geometry is evaluation-only.
2. Show random-init, task-trained, human-adapted, and sham stages together.
3. Make the paired task-trained to human-adapted comparison visually explicit.
4. Show the direct human-specific gain over sham with seed-level observations.
5. Show all four geometry components, including unfavorable shifts.

### Adaptation cost, retention, causality, and transfer

1. Show the gain-versus-parameter-displacement efficiency frontier.
2. Show the task-retention estimate against the fixed +/-0.02 gate.
3. Show causal specificity before and after adaptation.
4. Show cross-dataset transfer for every architecture and contrast.

The mock figures are marked as layout-only. Final rendering uses the same R
script with production CSV inputs and contains no mock watermark.

## Production rendering

After the six frozen production tables are present, render both vector and
300-dpi outputs with:

```powershell
& 'C:\Program Files\R\R-4.4.0\bin\Rscript.exe' scripts\render_unified_article_figures.R `
  --baseline-root publication-results `
  --adaptation-root results-extension\human-adaptation `
  --output results-extension\human-adaptation\figures-unified
```

The scheduled final-render check uses this command only after all required
tables exist. Mock mode is never used for production output.
