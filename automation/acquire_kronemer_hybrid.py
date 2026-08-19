#!/usr/bin/env python3
"""Acquire Kronemer EEG using full RP archives and selective ranges for large NRP tars."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import itertools
import json
import shutil
from pathlib import Path

from jacaccess.io.download import DownloadItem
from jacaccess.io.ranged_download import ranged_download

try:
    from automation.remote_tar_extract import (
        extract_remote,
        inspect_remote,
        load_or_create_index,
        sha256_file,
    )
    from automation.selective_tar_extract import extract as extract_local
except ModuleNotFoundError:  # Direct execution from the automation directory.
    from remote_tar_extract import (
        extract_remote,
        inspect_remote,
        load_or_create_index,
        sha256_file,
    )
    from selective_tar_extract import extract as extract_local


def read_manifest(path: Path) -> list[DownloadItem]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return [
            DownloadItem(
                url=row["url"],
                relative_path=row["relative_path"],
                expected_size_bytes=int(row["expected_size_bytes"]),
                sha256=row.get("sha256") or None,
            )
            for row in rows
        ]


def schedule_items(
    items: list[DownloadItem], download_root: Path, raw_root: Path, inventory_root: Path
) -> list[DownloadItem]:
    """Interleave network acquisition with local extraction and cheap no-ops.

    ThreadPoolExecutor starts submitted work FIFO.  A manifest ordered with all
    already-downloaded RP archives first would otherwise leave the network idle
    for hours while those archives are extracted.  Alternating the two queues
    overlaps network, CPU, and disk work without changing any scientific data.
    """
    complete: list[DownloadItem] = []
    network: list[DownloadItem] = []
    local: list[DownloadItem] = []
    for item in items:
        archive = download_root / Path(item.relative_path)
        name = archive.stem
        sentinel = raw_root / name / ".full_extraction_complete"
        receipt = inventory_root / f"{name}.json"
        if sentinel.exists() and receipt.exists():
            complete.append(item)
        elif archive.exists():
            local.append(item)
        else:
            network.append(item)

    ordered: list[DownloadItem] = complete
    for network_item, local_item in itertools.zip_longest(network, local):
        if network_item is not None:
            ordered.append(network_item)
        if local_item is not None:
            ordered.append(local_item)
    return ordered


def acquire_one(
    item: DownloadItem,
    *,
    download_root: Path,
    raw_root: Path,
    inventory_root: Path,
    index_root: Path,
    minimum_free_gib: float,
    timeout: float,
    retries: int,
    member_workers: int,
) -> dict[str, object]:
    relative = Path(item.relative_path)
    archive = download_root / relative
    name = archive.stem
    destination = raw_root / name
    sentinel = destination / ".full_extraction_complete"
    receipt = inventory_root / f"{name}.json"
    if sentinel.exists() and receipt.exists():
        return {"name": name, "mode": "existing", "status": "PASS"}
    free_gib = shutil.disk_usage(download_root).free / 2**30
    if free_gib < minimum_free_gib:
        raise RuntimeError(
            f"only {free_gib:.1f} GiB free before {name}; {minimum_free_gib:.1f} required"
        )
    if archive.exists() or name.endswith("_RP_EEG"):
        if not archive.exists():
            archive = ranged_download(
                item,
                download_root,
                connections=4,
                retries=retries,
                timeout=timeout,
            )
        extract_local(archive, destination, receipt)
        sentinel.touch()
        return {
            "name": name,
            "mode": "full_archive",
            "status": "PASS",
            "archive_bytes": archive.stat().st_size,
        }
    metadata = inspect_remote(item.url, item.expected_size_bytes, timeout)
    index_path = index_root / f"{name}.index.json"
    members, index_payload = load_or_create_index(
        metadata,
        index_path,
        block_bytes=4 * 1024,
        timeout=timeout,
        retries=retries,
        sparse=True,
    )
    extracted = extract_remote(
        metadata,
        members,
        destination,
        timeout=timeout,
        retries=retries,
        workers=member_workers,
    )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_suffix(receipt.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "status": "PASS",
                "mode": "selective_remote_ranges",
                "archive": item.url,
                "destination": str(destination),
                "metadata": metadata.__dict__,
                "index": str(index_path),
                "index_sha256": sha256_file(index_path),
                "selection_policy": index_payload["selection_policy"],
                "selected_count": len(members),
                "selected_bytes": sum(member.size_bytes for member in members),
                "archive_bytes_avoided": metadata.size_bytes
                - sum(member.size_bytes for member in members),
                "selected": extracted,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(receipt)
    sentinel.touch()
    return {
        "name": name,
        "mode": "selective_remote_ranges",
        "status": "PASS",
        "selected_bytes": sum(member.size_bytes for member in members),
        "archive_bytes": metadata.size_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--inventory-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--archive-workers", type=int, default=2)
    parser.add_argument("--member-workers", type=int, default=4)
    parser.add_argument("--minimum-free-gib", type=float, default=500)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()
    if args.archive_workers < 1 or args.member_workers < 1:
        parser.error("worker counts must be positive")
    items = schedule_items(
        read_manifest(args.manifest),
        args.download_root,
        args.raw_root,
        args.inventory_root,
    )
    kwargs = {
        "download_root": args.download_root,
        "raw_root": args.raw_root,
        "inventory_root": args.inventory_root,
        "index_root": args.index_root,
        "minimum_free_gib": args.minimum_free_gib,
        "timeout": args.timeout,
        "retries": args.retries,
        "member_workers": args.member_workers,
    }
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.archive_workers) as executor:
        futures = [executor.submit(acquire_one, item, **kwargs) for item in items]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(json.dumps(result, sort_keys=True), flush=True)
            results.append(result)
    if len(results) != len(items):
        raise RuntimeError("hybrid acquisition did not account for every manifest item")


if __name__ == "__main__":
    main()
