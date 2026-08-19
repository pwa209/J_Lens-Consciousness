"""Write a manuscript-ready factual digest from completed result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> None:
    theory = read("results/theory-comparison/theory-comparison.json")
    human = read("results/gates/human-production.json")
    machine = read("results/gates/machine-production-five-architectures.json")
    ranking = theory["architecture_ranking"]
    lines = [
        "# Ordinary article results digest",
        "",
        "This file is generated from completed, audited result artifacts.",
        "",
        "## Completion",
        "",
        f"- Human participants: {human['participants']}",
        f"- Completed human folds: {human['completed_summaries']}/{human['expected_folds']}",
        f"- Machine runs: {machine['completed_summaries']}/{machine['expected_runs']}",
        "",
        "## Five-theory comparison",
        "",
        f"The closest architecture was **{theory['winning_architecture']}**.",
        f"Capacity-limit interpretation falsified: **{theory['capacity_limit_interpretation_falsified']}**.",
        "",
        "| Rank | Architecture | Mean RMS distance | Mean cosine similarity |",
        "|---:|---|---:|---:|",
    ]
    for index, row in enumerate(ranking, start=1):
        lines.append(
            f"| {index} | {row['architecture']} | {row['mean_rms_distance']:.4f} | "
            f"{row['mean_cosine_similarity']:.4f} |"
        )
    lines.extend(["", "## Cross-dataset replication", ""])
    for contrast, result in theory["replication"].items():
        lines.append(
            f"- {contrast}: distance={result['standardized_distance_from_gabor_discovery']:.4f}; "
            f"same-direction fraction={result['same_direction_fraction']:.2f}; "
            f"n={result['participants']}."
        )
    output = ROOT / "results/article/results-digest.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

