"""Deterministic provenance records for every immutable workflow stage."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jacaccess.io.manifest import sha256_file


def git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def file_records(paths: Sequence[Path], root: Path) -> list[dict[str, Any]]:
    resolved_root = root.resolve()
    records: list[dict[str, Any]] = []
    for path in sorted((item.resolve() for item in paths), key=str):
        records.append(
            {
                "path": path.relative_to(resolved_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def provenance_record(
    *,
    root: Path,
    stage: str,
    command: Sequence[str],
    configuration_hashes: Mapping[str, str],
    inputs: Sequence[Path] = (),
    random_seeds: Sequence[int] = (),
    parent_artifacts: Sequence[Path] = (),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command),
        "working_directory": root.resolve().as_posix(),
        "git_commit": git_commit(root),
        "configuration_hashes": dict(sorted(configuration_hashes.items())),
        "random_seeds": list(random_seeds),
        "inputs": file_records(inputs, root),
        "parents": file_records(parent_artifacts, root),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["record_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def write_provenance(record: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)

