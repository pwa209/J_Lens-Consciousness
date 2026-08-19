"""Immutable SHA-256 manifests for source and derived artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    size_bytes: int
    sha256: str


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(paths: Iterable[Path], root: Path) -> list[ManifestEntry]:
    resolved_root = root.resolve()
    entries: list[ManifestEntry] = []
    for path in sorted((p.resolve() for p in paths), key=str):
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            relative = path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"{path} is outside manifest root {resolved_root}") from exc
        entries.append(
            ManifestEntry(
                path=relative.as_posix(),
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    return entries


def write_manifest(entries: Iterable[ManifestEntry], output: Path) -> None:
    payload = {
        "algorithm": "sha256",
        "files": [asdict(entry) for entry in entries],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def verify_manifest(entries: Iterable[ManifestEntry], root: Path) -> list[str]:
    failures: list[str] = []
    for entry in entries:
        path = root / entry.path
        if not path.is_file():
            failures.append(f"missing: {entry.path}")
            continue
        if path.stat().st_size != entry.size_bytes:
            failures.append(f"size changed: {entry.path}")
            continue
        if sha256_file(path) != entry.sha256:
            failures.append(f"hash changed: {entry.path}")
    return failures

