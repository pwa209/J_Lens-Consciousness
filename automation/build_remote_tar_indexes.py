#!/usr/bin/env python3
"""Build sparse indexes for NRP tar archives, optionally in parallel."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
from pathlib import Path

from remote_tar_extract import inspect_remote, load_or_create_index


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if Path(row["relative_path"]).name.endswith("_NRP.tar")
        ]


def build_one(
    row: dict[str, str],
    output: Path,
    timeout: float,
    retries: int,
) -> dict[str, object]:
    name = Path(row["relative_path"]).stem
    metadata = inspect_remote(row["url"], int(row["expected_size_bytes"]), timeout)
    index_path = output / f"{name}.index.json"
    members, payload = load_or_create_index(
        metadata,
        index_path,
        block_bytes=4 * 1024,
        timeout=timeout,
        retries=retries,
        sparse=True,
    )
    result = {
        "name": name,
        "index": str(index_path),
        "archive_bytes": metadata.size_bytes,
        "selected_bytes": sum(member.size_bytes for member in members),
        "avoided_bytes": metadata.size_bytes - sum(member.size_bytes for member in members),
        "index_transferred_bytes": int(payload.get("index_transferred_bytes", 0)),
        "selected_count": len(members),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.manifest)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(build_one, row, args.output, args.timeout, args.retries)
            for row in rows
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: str(item["name"]))
    payload = {
        "status": "PASS",
        "archive_count": len(results),
        "archive_bytes": sum(int(item["archive_bytes"]) for item in results),
        "selected_bytes": sum(int(item["selected_bytes"]) for item in results),
        "avoided_bytes": sum(int(item["avoided_bytes"]) for item in results),
        "index_transferred_bytes": sum(
            int(item["index_transferred_bytes"]) for item in results
        ),
        "archives": results,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.summary.with_suffix(args.summary.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.summary)


if __name__ == "__main__":
    main()
