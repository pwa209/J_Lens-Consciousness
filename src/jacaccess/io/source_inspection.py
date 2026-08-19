"""Read-only inspection of unknown EEG source trees before adapter verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SIGNAL_SUFFIXES = {
    ".set",
    ".edf",
    ".bdf",
    ".vhdr",
    ".cnt",
    ".fif",
    ".mat",
    ".raw",
    ".eeg",
}


def inspect_source_tree(participant_root: Path) -> dict[str, Any]:
    files = [path for path in participant_root.rglob("*") if path.is_file()]
    report: dict[str, Any] = {
        "root": participant_root.resolve().as_posix(),
        "file_count": len(files),
        "total_size_bytes": sum(path.stat().st_size for path in files),
        "signal_candidates": [],
        "tabular_files": [],
        "matlab_files": [],
    }
    for path in files:
        relative = path.relative_to(participant_root).as_posix()
        suffix = path.suffix.lower()
        if suffix in SIGNAL_SUFFIXES:
            report["signal_candidates"].append(
                {"path": relative, "suffix": suffix, "size_bytes": path.stat().st_size}
            )
        if suffix in {".tsv", ".csv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            try:
                header = path.open(encoding="utf-8-sig").readline().strip().split(delimiter)
            except UnicodeDecodeError:
                header = ["<non-utf8>"]
            report["tabular_files"].append({"path": relative, "columns": header})
        if suffix == ".mat":
            report["matlab_files"].append(_inspect_mat(path, participant_root))
    return report


def _inspect_mat(path: Path, root: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": path.relative_to(root).as_posix()}
    try:
        import scipy.io

        variables = scipy.io.whosmat(path)
        record["variables"] = [
            {"name": name, "shape": list(shape), "class": matlab_class}
            for name, shape, matlab_class in variables
        ]
    except (ImportError, NotImplementedError, ValueError, OSError) as exc:
        record["scipy_error"] = str(exc)
        try:
            import h5py

            with h5py.File(path, "r") as handle:
                record["hdf5_keys"] = sorted(handle.keys())
                record["hdf5_shapes"] = {
                    key: list(handle[key].shape)
                    for key in handle
                    if hasattr(handle[key], "shape")
                }
        except (ImportError, OSError) as hdf_exc:
            record["hdf5_error"] = str(hdf_exc)
    return record


def write_inspection_report(participant_root: Path, output: Path) -> None:
    report = inspect_source_tree(participant_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
