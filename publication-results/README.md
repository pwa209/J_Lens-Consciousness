# Public result subset

This directory is a disclosure-safe subset of the frozen production outputs.
It contains aggregate statistics, figures, machine summaries, quality-gate
summaries, and provenance records required to audit the article-level claims.

It deliberately excludes:

- raw or preprocessed EEG;
- participant- or trial-level metric tables;
- participant identifiers and fold-level warning labels;
- model checkpoints and intermediate tensors;
- download credentials, host paths, and runtime caches.

The complete 5.1 GB baseline snapshot and the 11.85 GB compressed adaptation
archive are stored offline on the project SSD with SHA-256 checksum files.
Access to any human-level derivatives should follow the licences and governance
requirements of the original data providers.

The primary main-text figure set is under `figures/science-advances-r/` and was
rendered natively with ggplot2 and patchwork. It contains six figures in PNG,
PDF, and SVG plus a manifest with SHA-256 hashes. The older
pre-adaptation exports are intentionally excluded from the public release and
are not the current article figure source.
