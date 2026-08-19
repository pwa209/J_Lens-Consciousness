"""Parallel HTTP range acquisition with safe resume and integrity receipts."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from jacaccess.io.download import (
    DownloadItem,
    read_download_manifest,
    resumable_http_download,
)

CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


@dataclass(frozen=True)
class RemoteMetadata:
    requested_url: str
    final_url: str
    size_bytes: int
    etag: str | None
    last_modified: str | None
    accepts_ranges: bool


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def inspect_remote(item: DownloadItem, timeout: float = 60) -> RemoteMetadata:
    response = requests.head(item.url, allow_redirects=True, timeout=timeout)
    response.raise_for_status()
    raw_size = response.headers.get("Content-Length")
    if raw_size is None:
        raise RuntimeError(f"server returned no Content-Length for {item.relative_path}")
    size = int(raw_size)
    if item.expected_size_bytes is not None and size != item.expected_size_bytes:
        raise ValueError(
            f"remote size {size} differs from manifest size "
            f"{item.expected_size_bytes} for {item.relative_path}"
        )
    return RemoteMetadata(
        requested_url=item.url,
        final_url=response.url,
        size_bytes=size,
        etag=response.headers.get("ETag"),
        last_modified=response.headers.get("Last-Modified"),
        accepts_ranges=response.headers.get("Accept-Ranges", "").lower() == "bytes",
    )


def split_ranges(start: int, stop: int, connections: int) -> list[tuple[int, int]]:
    """Split the half-open interval [start, stop) into inclusive HTTP ranges."""

    if start < 0 or stop < start:
        raise ValueError("invalid byte interval")
    if connections < 1:
        raise ValueError("connections must be positive")
    remaining = stop - start
    if remaining == 0:
        return []
    count = min(connections, remaining)
    base, extra = divmod(remaining, count)
    ranges = []
    cursor = start
    for index in range(count):
        length = base + int(index < extra)
        ranges.append((cursor, cursor + length - 1))
        cursor += length
    return ranges


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download_range(
    metadata: RemoteMetadata,
    start: int,
    end: int,
    path: Path,
    *,
    timeout: float,
    retries: int,
    chunk_bytes: int,
) -> None:
    expected = end - start + 1
    for attempt in range(1, retries + 1):
        existing = path.stat().st_size if path.exists() else 0
        if existing == expected:
            return
        if existing > expected:
            raise ValueError(f"range part {path} is larger than expected")
        current = start + existing
        headers = {"Range": f"bytes={current}-{end}"}
        if metadata.etag:
            headers["If-Range"] = metadata.etag
        try:
            with requests.get(
                metadata.final_url,
                headers=headers,
                stream=True,
                timeout=timeout,
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(
                        f"expected HTTP 206 for {current}-{end}, got {response.status_code}"
                    )
                match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
                if match is None:
                    raise RuntimeError("missing or invalid Content-Range response")
                received_start, received_end, received_total = map(int, match.groups())
                if (received_start, received_end, received_total) != (
                    current,
                    end,
                    metadata.size_bytes,
                ):
                    raise RuntimeError("server returned a different byte range than requested")
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("ab") as handle:
                    for chunk in response.iter_content(chunk_size=chunk_bytes):
                        if chunk:
                            handle.write(chunk)
            if path.stat().st_size == expected:
                return
            raise RuntimeError(f"incomplete range {start}-{end}")
        except (OSError, requests.RequestException, RuntimeError):
            if attempt == retries:
                raise
            time.sleep(min(30, 2**attempt))


def ranged_download(
    item: DownloadItem,
    destination_root: Path,
    *,
    connections: int = 8,
    retries: int = 4,
    timeout: float = 120,
    chunk_bytes: int = 4 * 1024 * 1024,
    keep_parts: bool = False,
) -> Path:
    metadata = inspect_remote(item, timeout=timeout)
    destination = destination_root / item.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = destination.with_name(destination.name + ".download.json")
    if destination.exists():
        if destination.stat().st_size != metadata.size_bytes:
            raise ValueError(f"existing final file has the wrong size: {destination}")
        previous = (
            json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else None
        )
        if previous is not None:
            if previous.get("metadata") != asdict(metadata):
                raise RuntimeError(f"remote metadata changed for completed {item.relative_path}")
            previous_digest = previous.get("sha256")
            if not previous_digest:
                raise RuntimeError(f"completed receipt lacks SHA-256 for {item.relative_path}")
            if item.sha256 is not None and previous_digest.lower() != item.sha256.lower():
                raise ValueError(f"completed receipt has the wrong SHA-256: {destination}")
            return destination
        digest = _sha256(destination)
        if item.sha256 is not None and digest.lower() != item.sha256.lower():
            raise ValueError(f"existing final file has the wrong SHA-256: {destination}")
        _atomic_json(
            receipt,
            {
                "status": "PASS",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "metadata": asdict(metadata),
                "relative_path": item.relative_path,
                "sha256": digest,
                "adopted_existing_file": True,
            },
        )
        return destination

    if not metadata.accepts_ranges or connections == 1:
        downloaded = resumable_http_download(item, destination_root, chunk_bytes=chunk_bytes)
        digest = _sha256(downloaded)
        _atomic_json(
            receipt,
            {
                "status": "PASS",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "metadata": asdict(metadata),
                "relative_path": item.relative_path,
                "sha256": digest,
                "mode": "single_stream_fallback",
            },
        )
        return downloaded

    prefix = destination.with_suffix(destination.suffix + ".part")
    work = destination.with_name(destination.name + ".ranges")
    state_path = work / "state.json"
    state: dict[str, Any]
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["metadata"] != asdict(metadata):
            raise RuntimeError(f"remote metadata changed while resuming {item.relative_path}")
        prefix_bytes = int(state["prefix_bytes"])
        if (prefix.stat().st_size if prefix.exists() else 0) != prefix_bytes:
            raise RuntimeError("sequential prefix size changed after range state was created")
        ranges = [(int(start), int(end)) for start, end in state["ranges"]]
    else:
        prefix_bytes = prefix.stat().st_size if prefix.exists() else 0
        if prefix_bytes > metadata.size_bytes:
            raise ValueError("sequential partial is larger than the remote object")
        ranges = split_ranges(prefix_bytes, metadata.size_bytes, connections)
        state = {
            "metadata": asdict(metadata),
            "prefix_bytes": prefix_bytes,
            "ranges": ranges,
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_json(state_path, state)

    def transfer(byte_range: tuple[int, int]) -> None:
        start, end = byte_range
        _download_range(
            metadata,
            start,
            end,
            work / f"range-{start}-{end}.part",
            timeout=timeout,
            retries=retries,
            chunk_bytes=chunk_bytes,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=connections) as executor:
        futures = [executor.submit(transfer, byte_range) for byte_range in ranges]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    assembling = destination.with_name(destination.name + ".assembling")
    digest = hashlib.sha256()
    total_written = 0
    with assembling.open("wb") as output:
        sources = ([prefix] if prefix_bytes else []) + [
            work / f"range-{start}-{end}.part" for start, end in ranges
        ]
        for source in sources:
            with source.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    total_written += len(chunk)
    encoded_digest = digest.hexdigest()
    if total_written != metadata.size_bytes:
        raise RuntimeError("assembled file size differs from remote Content-Length")
    if item.sha256 is not None and encoded_digest.lower() != item.sha256.lower():
        raise RuntimeError("assembled file SHA-256 differs from the manifest")
    assembling.replace(destination)
    _atomic_json(
        receipt,
        {
            "status": "PASS",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "metadata": asdict(metadata),
            "relative_path": item.relative_path,
            "sha256": encoded_digest,
            "mode": "parallel_ranges",
            "connections": connections,
            "adopted_prefix_bytes": prefix_bytes,
            "ranges": ranges,
        },
    )
    if not keep_parts:
        if prefix.exists():
            prefix.unlink()
        for start, end in ranges:
            (work / f"range-{start}-{end}.part").unlink()
        state_path.unlink()
        work.rmdir()
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument(
        "--item-workers",
        type=int,
        default=1,
        help="Number of independent manifest items to download concurrently.",
    )
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--minimum-free-gib", type=float, default=0)
    parser.add_argument("--keep-parts", action="store_true")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    if args.item_workers < 1:
        parser.error("--item-workers must be positive")
    items = read_download_manifest(args.manifest)

    def acquire(item: DownloadItem) -> str:
        free_gib = shutil.disk_usage(args.destination).free / 2**30
        if free_gib < args.minimum_free_gib:
            raise RuntimeError(
                f"only {free_gib:.1f} GiB free before {item.relative_path}; "
                f"{args.minimum_free_gib:.1f} GiB required"
            )
        path = ranged_download(
            item,
            args.destination,
            connections=args.connections,
            retries=args.retries,
            timeout=args.timeout,
            keep_parts=args.keep_parts,
        )
        print(f"completed: {path}", flush=True)
        return str(path)

    if args.item_workers == 1:
        completed = [acquire(item) for item in items]
    else:
        completed = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.item_workers
        ) as executor:
            futures = [executor.submit(acquire, item) for item in items]
            for future in concurrent.futures.as_completed(futures):
                completed.append(future.result())
    print(json.dumps({"completed": completed}, indent=2))


if __name__ == "__main__":
    main()
