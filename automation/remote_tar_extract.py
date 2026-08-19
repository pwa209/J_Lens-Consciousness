#!/usr/bin/env python3
"""Index an uncompressed remote tar and fetch only selected members by HTTP range."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import re
import tarfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

try:
    from automation.selective_tar_extract import _safe_relative, _selected
except ModuleNotFoundError:  # Direct execution sets automation/ as sys.path[0].
    from selective_tar_extract import _safe_relative, _selected

CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
SPARSE_SKIP_DIRECTORY_NAMES = {"mri_data", "dicom", "dti", "fmri", "nifti", "pet", "video"}


@dataclass(frozen=True)
class RemoteMetadata:
    requested_url: str
    final_url: str
    size_bytes: int
    etag: str | None
    last_modified: str | None
    accepts_ranges: bool


@dataclass(frozen=True)
class SelectedMember:
    path: str
    size_bytes: int
    offset_data: int
    mtime: float
    mode: int


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def inspect_remote(url: str, expected_size: int | None, timeout: float) -> RemoteMetadata:
    response = requests.head(url, allow_redirects=True, timeout=timeout)
    response.raise_for_status()
    raw_size = response.headers.get("Content-Length")
    if raw_size is None:
        raise RuntimeError("remote archive has no Content-Length")
    size = int(raw_size)
    if expected_size is not None and size != expected_size:
        raise ValueError(f"remote archive size {size} differs from expected {expected_size}")
    accepts_ranges = response.headers.get("Accept-Ranges", "").lower() == "bytes"
    if not accepts_ranges:
        raise RuntimeError("remote archive does not advertise HTTP byte ranges")
    return RemoteMetadata(
        requested_url=url,
        final_url=response.url,
        size_bytes=size,
        etag=response.headers.get("ETag"),
        last_modified=response.headers.get("Last-Modified"),
        accepts_ranges=accepts_ranges,
    )


class HTTPRangeReader(io.RawIOBase):
    """Small seekable reader for tar metadata traversal over HTTP ranges."""

    def __init__(
        self,
        metadata: RemoteMetadata,
        *,
        block_bytes: int = 4 * 1024,
        timeout: float = 60,
        retries: int = 6,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.block_bytes = block_bytes
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()
        self.position = 0
        self.cache_start = 0
        self.cache = b""
        self.request_count = 0
        self.transferred_bytes = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.metadata.size_bytes + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = position
        return self.position

    def _cached(self, start: int, length: int) -> bytes | None:
        relative = start - self.cache_start
        if relative < 0 or relative + length > len(self.cache):
            return None
        return self.cache[relative : relative + length]

    def _fetch(self, start: int, minimum_bytes: int) -> None:
        end = min(
            self.metadata.size_bytes - 1,
            start + max(minimum_bytes, self.block_bytes) - 1,
        )
        headers = {"Range": f"bytes={start}-{end}"}
        if self.metadata.etag:
            headers["If-Range"] = self.metadata.etag
        for attempt in range(1, self.retries + 1):
            try:
                response = self.session.get(
                    self.metadata.final_url,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"expected HTTP 206, got {response.status_code}")
                match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
                if match is None:
                    raise RuntimeError("missing or invalid Content-Range")
                received_start, received_end, received_total = map(int, match.groups())
                if (received_start, received_end, received_total) != (
                    start,
                    end,
                    self.metadata.size_bytes,
                ):
                    raise RuntimeError("server returned an unexpected byte range")
                if len(response.content) != end - start + 1:
                    raise RuntimeError("range response body has the wrong length")
                self.cache_start = start
                self.cache = response.content
                self.request_count += 1
                self.transferred_bytes += len(response.content)
                return
            except (OSError, requests.RequestException, RuntimeError):
                if attempt == self.retries:
                    raise
                time.sleep(min(30, 2**attempt))

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.metadata.size_bytes:
            return b""
        if size is None or size < 0:
            size = self.metadata.size_bytes - self.position
        size = min(size, self.metadata.size_bytes - self.position)
        cached = self._cached(self.position, size)
        if cached is None:
            self._fetch(self.position, size)
            cached = self._cached(self.position, size)
        if cached is None:
            raise RuntimeError("internal range-cache error")
        self.position += len(cached)
        return cached

    def read_at(self, offset: int, size: int) -> bytes:
        previous = self.position
        try:
            self.seek(offset)
            return self.read(size)
        finally:
            self.position = previous


def selected_from_tar(handle: tarfile.TarFile) -> tuple[list[SelectedMember], int]:
    selected: list[SelectedMember] = []
    skipped = 0
    for member in handle:
        if not member.isfile():
            continue
        if not _selected(member.name):
            skipped += 1
            continue
        relative = _safe_relative(member.name)
        selected.append(
            SelectedMember(
                path=relative.as_posix(),
                size_bytes=member.size,
                offset_data=member.offset_data,
                mtime=float(member.mtime),
                mode=int(member.mode),
            )
        )
    return selected, skipped


def _parse_header(block: bytes, offset: int, root: str | None = None) -> tarfile.TarInfo | None:
    if len(block) != tarfile.BLOCKSIZE or block == b"\0" * tarfile.BLOCKSIZE:
        return None
    try:
        member = tarfile.TarInfo.frombuf(block, "utf-8", "surrogateescape")
    except (tarfile.HeaderError, UnicodeError, ValueError):
        return None
    if member.type in {tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK}:
        member.offset = offset
        member.offset_data = offset + tarfile.BLOCKSIZE
        return member
    try:
        _safe_relative(member.name)
    except ValueError:
        return None
    if root is not None and member.name != root and not member.name.startswith(root + "/"):
        return None
    member.offset = offset
    member.offset_data = offset + tarfile.BLOCKSIZE
    return member


def _next_tar_offset(member: tarfile.TarInfo) -> int:
    blocks = (member.size + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE
    return member.offset_data + blocks * tarfile.BLOCKSIZE


def _read_member_at(
    reader: HTTPRangeReader,
    offset: int,
    root: str | None,
) -> tuple[tarfile.TarInfo | None, int]:
    member = _parse_header(reader.read_at(offset, tarfile.BLOCKSIZE), offset)
    if member is None:
        return None, offset
    if member.type == tarfile.GNUTYPE_LONGNAME:
        payload = reader.read_at(member.offset_data, member.size)
        long_name = payload.rstrip(b"\0").decode("utf-8", "surrogateescape")
        actual_offset = _next_tar_offset(member)
        actual = _parse_header(
            reader.read_at(actual_offset, tarfile.BLOCKSIZE),
            actual_offset,
        )
        if actual is None:
            raise RuntimeError("GNU LongLink was not followed by a valid tar header")
        actual.name = long_name
        member = actual
    if member.type == tarfile.GNUTYPE_LONGLINK:
        raise RuntimeError("GNU long link targets are unsupported in regular-file selection")
    try:
        _safe_relative(member.name)
    except ValueError:
        return None, offset
    if root is not None and member.name != root and not member.name.startswith(root + "/"):
        return None, offset
    return member, _next_tar_offset(member)


def _probe_header(
    reader: HTTPRangeReader,
    start: int,
    stop: int,
    *,
    root: str,
    window_bytes: int,
) -> tarfile.TarInfo | None:
    aligned = max(0, start - (start % tarfile.BLOCKSIZE))
    length = min(window_bytes, max(0, stop - aligned))
    if length < tarfile.BLOCKSIZE:
        return None
    payload = reader.read_at(aligned, length)
    usable = len(payload) - (len(payload) % tarfile.BLOCKSIZE)
    for relative in range(0, usable, tarfile.BLOCKSIZE):
        member = _parse_header(
            payload[relative : relative + tarfile.BLOCKSIZE],
            aligned + relative,
            root,
        )
        if member is not None and member.type not in {
            tarfile.GNUTYPE_LONGNAME,
            tarfile.GNUTYPE_LONGLINK,
        }:
            return member
    return None


def _find_subtree_end(
    reader: HTTPRangeReader,
    start: int,
    prefix: str,
    *,
    root: str,
    probe_bytes: int,
) -> int:
    """Find the first tar header outside a contiguous directory subtree."""

    low = start
    high = reader.metadata.size_bytes
    best_outside: int | None = None
    while high - low > probe_bytes:
        midpoint = ((low + high) // 2) // tarfile.BLOCKSIZE * tarfile.BLOCKSIZE
        member = _probe_header(
            reader,
            midpoint,
            high,
            root=root,
            window_bytes=probe_bytes,
        )
        if member is None:
            # A missing header is expected only near the terminating zero blocks.
            high = midpoint
        elif member.name.startswith(prefix):
            low = max(low + tarfile.BLOCKSIZE, member.offset + tarfile.BLOCKSIZE)
        else:
            best_outside = member.offset
            high = member.offset
    scan_stop = min(reader.metadata.size_bytes, max(high, low + probe_bytes))
    payload = reader.read_at(low, scan_stop - low)
    usable = len(payload) - (len(payload) % tarfile.BLOCKSIZE)
    for relative in range(0, usable, tarfile.BLOCKSIZE):
        member = _parse_header(
            payload[relative : relative + tarfile.BLOCKSIZE],
            low + relative,
            root,
        )
        if member is not None and not member.name.startswith(prefix):
            return member.offset
    return best_outside if best_outside is not None else reader.metadata.size_bytes


def index_remote_sparse(
    metadata: RemoteMetadata,
    *,
    timeout: float,
    retries: int,
    probe_bytes: int = 4 * 1024 * 1024,
) -> tuple[list[SelectedMember], dict[str, int]]:
    reader = HTTPRangeReader(
        metadata,
        block_bytes=4 * 1024,
        timeout=timeout,
        retries=retries,
    )
    first, _ = _read_member_at(reader, 0, None)
    if first is None:
        raise RuntimeError("remote archive does not begin with a valid POSIX tar header")
    root = first.name.rstrip("/").split("/", 1)[0]
    selected: list[SelectedMember] = []
    skipped = 0
    skipped_subtrees = 0
    offset = 0
    while offset + tarfile.BLOCKSIZE <= metadata.size_bytes:
        member, next_offset = _read_member_at(reader, offset, root)
        if member is None:
            raw = reader.read_at(offset, tarfile.BLOCKSIZE)
            if raw != b"\0" * tarfile.BLOCKSIZE:
                raise RuntimeError(
                    f"unsupported or invalid nonzero tar header at byte offset {offset}"
                )
            break
        relative = _safe_relative(member.name)
        if member.isdir() and relative.name.lower() in SPARSE_SKIP_DIRECTORY_NAMES:
            prefix = member.name.rstrip("/") + "/"
            offset = _find_subtree_end(
                reader,
                _next_tar_offset(member),
                prefix,
                root=root,
                probe_bytes=probe_bytes,
            )
            skipped_subtrees += 1
            continue
        if member.isfile():
            if _selected(member.name):
                selected.append(
                    SelectedMember(
                        path=relative.as_posix(),
                        size_bytes=member.size,
                        offset_data=member.offset_data,
                        mtime=float(member.mtime),
                        mode=int(member.mode),
                    )
                )
            else:
                skipped += 1
        offset = next_offset
    if not selected:
        raise RuntimeError("sparse selection found no files in remote archive")
    return selected, {
        "skipped_file_count": skipped,
        "skipped_subtree_count": skipped_subtrees,
        "index_request_count": reader.request_count,
        "index_transferred_bytes": reader.transferred_bytes,
    }


def index_remote(
    metadata: RemoteMetadata,
    *,
    block_bytes: int,
    timeout: float,
    retries: int,
) -> tuple[list[SelectedMember], dict[str, int]]:
    reader = HTTPRangeReader(
        metadata,
        block_bytes=block_bytes,
        timeout=timeout,
        retries=retries,
    )
    with tarfile.open(fileobj=reader, mode="r:") as handle:
        selected, skipped = selected_from_tar(handle)
    if not selected:
        raise RuntimeError("selection found no files in remote archive")
    return selected, {
        "skipped_file_count": skipped,
        "index_request_count": reader.request_count,
        "index_transferred_bytes": reader.transferred_bytes,
    }


def index_local(archive: Path) -> tuple[list[SelectedMember], int]:
    with tarfile.open(archive, mode="r:*") as handle:
        return selected_from_tar(handle)


def load_or_create_index(
    metadata: RemoteMetadata,
    index_path: Path,
    *,
    block_bytes: int,
    timeout: float,
    retries: int,
    sparse: bool = False,
) -> tuple[list[SelectedMember], dict[str, Any]]:
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if payload.get("metadata") != asdict(metadata):
            raise RuntimeError("remote archive metadata changed after index creation")
        members = [SelectedMember(**member) for member in payload["selected"]]
        return members, payload
    if sparse:
        members, statistics = index_remote_sparse(
            metadata,
            timeout=timeout,
            retries=retries,
        )
    else:
        members, statistics = index_remote(
            metadata,
            block_bytes=block_bytes,
            timeout=timeout,
            retries=retries,
        )
    payload: dict[str, Any] = {
        "status": "PASS",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "metadata": asdict(metadata),
        "selection_policy": "eeg_and_metadata_suffixes_excluding_imaging_and_video",
        "index_mode": "sparse_subtree_skip" if sparse else "standard_tar_walk",
        "selected_count": len(members),
        "selected_bytes": sum(member.size_bytes for member in members),
        "selected": [asdict(member) for member in members],
        **statistics,
    }
    atomic_json(index_path, payload)
    return members, payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_member(
    metadata: RemoteMetadata,
    member: SelectedMember,
    destination: Path,
    *,
    timeout: float,
    retries: int,
    chunk_bytes: int = 4 * 1024 * 1024,
) -> dict[str, Any]:
    relative = _safe_relative(member.path)
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".remote.part")
    if target.exists():
        if target.stat().st_size != member.size_bytes:
            raise ValueError(f"existing extracted file has wrong size: {target}")
        return {
            "path": member.path,
            "size_bytes": member.size_bytes,
            "sha256": sha256_file(target),
            "adopted_existing_file": True,
        }
    if partial.exists() and partial.stat().st_size > member.size_bytes:
        raise ValueError(f"partial extracted file is too large: {partial}")
    for attempt in range(1, retries + 1):
        existing = partial.stat().st_size if partial.exists() else 0
        if existing == member.size_bytes:
            break
        start = member.offset_data + existing
        end = member.offset_data + member.size_bytes - 1
        headers = {"Range": f"bytes={start}-{end}"}
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
                    raise RuntimeError(f"expected HTTP 206, got {response.status_code}")
                match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
                if match is None:
                    raise RuntimeError("missing or invalid Content-Range")
                received_start, received_end, received_total = map(int, match.groups())
                if (received_start, received_end, received_total) != (
                    start,
                    end,
                    metadata.size_bytes,
                ):
                    raise RuntimeError("server returned an unexpected member byte range")
                with partial.open("ab") as output:
                    for chunk in response.iter_content(chunk_size=chunk_bytes):
                        if chunk:
                            output.write(chunk)
        except (OSError, requests.RequestException, RuntimeError):
            if attempt == retries:
                raise
            time.sleep(min(30, 2**attempt))
    if partial.stat().st_size != member.size_bytes:
        raise RuntimeError(f"remote member download is incomplete: {member.path}")
    partial.replace(target)
    try:
        target.chmod(member.mode & 0o777)
    except OSError:
        pass
    return {
        "path": member.path,
        "size_bytes": member.size_bytes,
        "sha256": sha256_file(target),
        "adopted_existing_file": False,
    }


def extract_remote(
    metadata: RemoteMetadata,
    members: list[SelectedMember],
    destination: Path,
    *,
    timeout: float,
    retries: int,
    workers: int,
) -> list[dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                download_member,
                metadata,
                member,
                destination,
                timeout=timeout,
                retries=retries,
            )
            for member in members
        ]
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(f"remote member complete: {result['path']}", flush=True)
            results.append(result)
    return sorted(results, key=lambda item: str(item["path"]))


def validate_against_local(
    metadata: RemoteMetadata,
    remote_members: list[SelectedMember],
    local_archive: Path,
    *,
    timeout: float,
    retries: int,
    sample_bytes: int = 64 * 1024,
) -> dict[str, Any]:
    local_members, local_skipped = index_local(local_archive)
    if remote_members != local_members:
        raise RuntimeError("remote and local selected tar indexes differ")
    reader = HTTPRangeReader(
        metadata,
        block_bytes=sample_bytes,
        timeout=timeout,
        retries=retries,
    )
    comparisons = 0
    compared_bytes = 0
    with local_archive.open("rb") as local:
        for member in remote_members:
            if member.size_bytes == 0:
                continue
            length = min(sample_bytes, member.size_bytes)
            offsets = {0, max(0, (member.size_bytes - length) // 2), member.size_bytes - length}
            for relative_offset in sorted(offsets):
                absolute = member.offset_data + relative_offset
                local.seek(absolute)
                expected = local.read(length)
                observed = reader.read_at(absolute, length)
                if hashlib.sha256(expected).digest() != hashlib.sha256(observed).digest():
                    raise RuntimeError(f"remote byte mismatch for {member.path}")
                comparisons += 1
                compared_bytes += length
    return {
        "status": "PASS",
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "local_archive": str(local_archive),
        "metadata": asdict(metadata),
        "selected_count": len(remote_members),
        "local_skipped_file_count": local_skipped,
        "sample_comparisons": comparisons,
        "sampled_bytes": compared_bytes,
        "sample_request_count": reader.request_count,
        "sample_transferred_bytes": reader.transferred_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--expected-size", type=int)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--validate-local-archive", type=Path)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--verify-index-local-archive", type=Path)
    parser.add_argument("--verify-output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--index-block-bytes", type=int, default=4 * 1024)
    parser.add_argument("--sparse-index", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 1 or args.index_block_bytes < 512:
        parser.error("worker, retry, and index-block settings must be positive")
    if args.verify_index_local_archive is not None:
        if args.verify_output is None:
            parser.error("--verify-output is required with --verify-index-local-archive")
        payload = json.loads(args.index.read_text(encoding="utf-8"))
        indexed = [SelectedMember(**member) for member in payload["selected"]]
        local, local_skipped = index_local(args.verify_index_local_archive)
        if indexed != local:
            raise RuntimeError("saved remote index differs from local full archive index")
        atomic_json(
            args.verify_output,
            {
                "status": "PASS",
                "verified_at_utc": datetime.now(UTC).isoformat(),
                "index": str(args.index),
                "index_sha256": sha256_file(args.index),
                "local_archive": str(args.verify_index_local_archive),
                "selected_count": len(indexed),
                "selected_bytes": sum(member.size_bytes for member in indexed),
                "local_skipped_file_count": local_skipped,
                "comparison": "exact_path_size_offset_mtime_mode_equality",
            },
        )
        return
    if args.url is None:
        parser.error("--url is required except for local index verification")
    metadata = inspect_remote(args.url, args.expected_size, args.timeout)
    members, index_payload = load_or_create_index(
        metadata,
        args.index,
        block_bytes=args.index_block_bytes,
        timeout=args.timeout,
        retries=args.retries,
        sparse=args.sparse_index,
    )
    print(
        json.dumps(
            {
                "selected_count": len(members),
                "selected_bytes": sum(member.size_bytes for member in members),
                "archive_bytes": metadata.size_bytes,
                "selection_fraction": sum(member.size_bytes for member in members)
                / metadata.size_bytes,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.validate_local_archive is not None:
        validation = validate_against_local(
            metadata,
            members,
            args.validate_local_archive,
            timeout=args.timeout,
            retries=args.retries,
        )
        if args.validation_output is None:
            parser.error("--validation-output is required with --validate-local-archive")
        atomic_json(args.validation_output, validation)
    if args.index_only:
        return
    if args.destination is None or args.receipt is None:
        parser.error("--destination and --receipt are required for extraction")
    extracted = extract_remote(
        metadata,
        members,
        args.destination,
        timeout=args.timeout,
        retries=args.retries,
        workers=args.workers,
    )
    atomic_json(
        args.receipt,
        {
            "status": "PASS",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "metadata": asdict(metadata),
            "index": str(args.index),
            "index_sha256": sha256_file(args.index),
            "selection_policy": index_payload["selection_policy"],
            "selected_count": len(members),
            "selected_bytes": sum(member.size_bytes for member in members),
            "extracted": extracted,
        },
    )


if __name__ == "__main__":
    main()
