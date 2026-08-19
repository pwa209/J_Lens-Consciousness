"""Build the production participant roster from verified acquired data."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def _firewalled(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["dataset_id"], row["participant_id"])
            for row in csv.DictReader(handle, delimiter="\t")
            if str(row.get("confirmatory_include", "")).strip().lower()
            not in {"1", "true", "yes"}
        }


def _brainvision_is_complete(path: Path) -> bool:
    text = path.read_text(encoding="latin-1")
    references = []
    for key in ("DataFile", "MarkerFile"):
        match = re.search(rf"^{key}=(.+)$", text, flags=re.MULTILINE)
        if match is None:
            return False
        references.append(path.parent / match.group(1).strip())
    return all(reference.exists() for reference in references)


def _gabor(root: Path) -> list[tuple[str, str, str]]:
    rows = []
    for path in sorted(root.glob("sub-*")):
        if not path.is_dir() or not any((path / "eeg").glob("*_events.tsv")):
            continue
        signals = [
            candidate
            for candidate in (path / "eeg").iterdir()
            if candidate.suffix.lower() in {".vhdr", ".set", ".edf"}
        ]
        if not signals:
            continue
        complete = any(
            candidate.suffix.lower() != ".vhdr"
            or _brainvision_is_complete(candidate)
            for candidate in signals
        )
        rows.append(
            (
                path.name,
                "1" if complete else "0",
                "full acquisition inventory"
                if complete
                else "source exclusion: incomplete BrainVision file set",
            )
        )
    return rows


def _somato(root: Path) -> list[str]:
    dataset = root / "DATASET_PREPROCESSED"
    return [
        path.name
        for path in sorted(dataset.glob("sub*"), key=lambda item: int(item.name[3:]))
        if path.is_dir() and any(path.rglob("*.mat"))
    ]


def _kronemer(root: Path) -> list[tuple[str, str, str]]:
    pattern = re.compile(r"^\d+_(?:RP_EEG|NRP)$")
    rows = []
    for path in sorted(root.iterdir()):
        if not (
            path.is_dir()
            and pattern.fullmatch(path.name)
            and (
                (path / ".full_extraction_complete").exists()
                or (path / ".pilot_extraction_complete").exists()
            )
        ):
            continue
        task_raws = [
            candidate
            for candidate in path.rglob("*.raw")
            if "calibration" not in candidate.as_posix().lower()
            and "/EEG_Session/" in candidate.as_posix()
        ]
        rows.append(
            (
                path.name,
                "1" if task_raws else "0",
                "full acquisition inventory"
                if task_raws
                else "source exclusion: no task EEG recording",
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/execution/participants.tsv"),
    )
    parser.add_argument(
        "--pilot-firewall",
        type=Path,
        default=Path("configs/execution/pilot_firewall.tsv"),
    )
    args = parser.parse_args()
    excluded = _firewalled(args.pilot_firewall)
    rows = [
        ("gabor", participant, include, reason)
        for participant, include, reason in _gabor(args.raw_root / "gabor")
        if ("gabor", participant) not in excluded
    ]
    rows.extend(
        ("somato", participant, "1", "full archive inventory")
        for participant in _somato(args.raw_root / "somato")
        if ("somato", participant) not in excluded
    )
    rows.extend(
        ("kronemer", participant, include, reason)
        for participant, include, reason in _kronemer(args.raw_root / "kronemer")
        if ("kronemer", participant) not in excluded
    )
    if not any(row[0] == "gabor" and row[2] == "1" for row in rows):
        raise SystemExit("no acquired Gabor participants were found")
    if not any(row[0] == "somato" and row[2] == "1" for row in rows):
        raise SystemExit("no acquired Somato participants were found")
    if not any(row[0] == "kronemer" and row[2] == "1" for row in rows):
        raise SystemExit("no acquired Kronemer participants were found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("dataset_id", "participant_id", "include", "reason"))
        writer.writerows(rows)
    temporary.replace(args.output)
    print(
        " ".join(
            f"{dataset}={sum(row[0] == dataset and row[2] == '1' for row in rows)}"
            for dataset in ("gabor", "somato", "kronemer")
        )
    )


if __name__ == "__main__":
    main()
