"""Safely extract likely EEG and metadata members from Kronemer tar archives."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path, PurePosixPath

EEG_AND_METADATA_SUFFIXES = {
    ".bdf",
    ".bin",
    ".ced",
    ".cnt",
    ".csv",
    ".dat",
    ".edf",
    ".eeg",
    ".elp",
    ".fdt",
    ".json",
    ".loc",
    ".mat",
    ".raw",
    ".set",
    ".sfp",
    ".tsv",
    ".txt",
    ".vhdr",
    ".vmrk",
    ".xml",
}
EXCLUDED_PARTS = {
    "dicom",
    "dti",
    "fmri",
    "mri",
    "mri_data",
    "nifti",
    "pet",
    "t1",
    "t2",
    "video",
}


def _safe_relative(name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe archive member: {name}")
    return Path(*pure.parts)


def _selected(name: str) -> bool:
    relative = _safe_relative(name)
    lowered = {part.lower() for part in relative.parts}
    if lowered & EXCLUDED_PARTS:
        return False
    return relative.suffix.lower() in EEG_AND_METADATA_SUFFIXES


def extract(archive: Path, destination: Path, receipt: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, object]] = []
    skipped = 0
    with tarfile.open(archive, "r:*") as handle:
        for member in handle:
            if not member.isfile():
                continue
            relative = _safe_relative(member.name)
            if not _selected(member.name):
                skipped += 1
                continue
            source = handle.extractfile(member)
            if source is None:
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            with source, temporary.open("wb") as output:
                while chunk := source.read(8 * 1024 * 1024):
                    output.write(chunk)
            temporary.replace(target)
            selected.append({"path": relative.as_posix(), "size_bytes": member.size})
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps(
            {
                "archive": str(archive),
                "destination": str(destination),
                "selection_policy": "eeg_and_metadata_suffixes_excluding_imaging_and_video",
                "selected_count": len(selected),
                "skipped_file_count": skipped,
                "selected": selected,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if not selected:
        raise RuntimeError(f"selection extracted no files from {archive}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    extract(args.archive, args.destination, args.receipt)


if __name__ == "__main__":
    main()
