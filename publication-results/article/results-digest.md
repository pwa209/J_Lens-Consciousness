# Ordinary article results digest

This file is generated from completed, audited result artifacts.

## Completion

- Human participants: 173
- Completed human folds: 865/865
- Machine runs: 100/100

## Five-theory comparison

The closest architecture was **feedforward**.
Capacity-limit interpretation falsified: **False**.

| Rank | Architecture | Mean RMS distance | Mean cosine similarity |
|---:|---|---:|---:|
| 1 | feedforward | 0.4317 | 0.7650 |
| 2 | shared_workspace | 0.4584 | 0.3312 |
| 3 | unlimited_shared_state | 0.4712 | 0.3278 |
| 4 | private_modules | 0.4826 | 0.2278 |
| 5 | recurrent | 0.6203 | -0.0423 |

## Cross-dataset replication

- gabor-0: distance=0.0000; same-direction fraction=1.00; n=28.
- kronemer-0: distance=0.5268; same-direction fraction=0.75; n=116.
- kronemer-1: distance=0.3634; same-direction fraction=0.75; n=116.
- somato-0: distance=0.9251; same-direction fraction=0.75; n=29.
- somato-1: distance=2.2431; same-direction fraction=0.50; n=29.

## Human-guided adaptation

- Completed outer-fold runs: 1,000/1,000.
- Retention-gate failures: 34/1,000, balanced between human-target and sham
  conditions (17 each).
- Holm-corrected held-out adaptation benefits: none.
- Mean human-adapted change from task training: feedforward -0.00747,
  recurrent -0.10875, private modules +0.01517, capacity-limited shared state
  -0.07888, and unlimited shared state -0.09528 (all Holm p = 1).
- The capacity-limited all-runs sham contrast was nominally positive
  (p = 0.0281) but not multiplicity-corrected (Holm p = 0.1405) and not robust
  in the complete-case analysis (p = 0.0587; Holm p = 0.293).

The adaptation stage therefore supplies a constraint and falsification result,
not evidence that human-neural fine-tuning improves held-out geometry.
