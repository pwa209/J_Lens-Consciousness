"""Resumable authenticated/public downloads followed by immutable manifests."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from jacaccess.io.manifest import build_manifest, sha256_file, write_manifest


@dataclass(frozen=True)
class DownloadItem:
    url: str
    relative_path: str
    expected_size_bytes: int | None = None
    sha256: str | None = None


def read_download_manifest(path: Path) -> list[DownloadItem]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    items: list[DownloadItem] = []
    for row in rows:
        if not row.get("url"):
            continue
        items.append(
            DownloadItem(
                url=row["url"],
                relative_path=row["relative_path"],
                expected_size_bytes=(
                    int(row["expected_size_bytes"]) if row.get("expected_size_bytes") else None
                ),
                sha256=row.get("sha256") or None,
            )
        )
    return items


def resumable_http_download(
    item: DownloadItem,
    destination_root: Path,
    *,
    cookie: str | None = None,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> Path:
    destination = destination_root / item.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if (
            item.expected_size_bytes is not None
            and destination.stat().st_size != item.expected_size_bytes
        ):
            raise ValueError(f"existing size differs for {item.relative_path}")
        if item.sha256 is not None and sha256_file(destination).lower() != item.sha256.lower():
            raise ValueError(f"existing SHA-256 differs for {item.relative_path}")
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    if cookie:
        headers["Cookie"] = cookie
    with requests.get(item.url, headers=headers, stream=True, timeout=120) as response:
        if existing and response.status_code != 206:
            existing = 0
            partial.unlink(missing_ok=True)
        response.raise_for_status()
        mode = "ab" if existing else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=chunk_bytes):
                if chunk:
                    handle.write(chunk)
    partial.replace(destination)
    if item.expected_size_bytes is not None and destination.stat().st_size != item.expected_size_bytes:
        raise ValueError(f"downloaded size differs for {item.relative_path}")
    if item.sha256 is not None and sha256_file(destination).lower() != item.sha256.lower():
        raise ValueError(f"SHA-256 differs for {item.relative_path}")
    return destination


def acquire_openneuro(dataset_id: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            "--no-sign-request",
            f"s3://openneuro.org/{dataset_id}",
            str(destination),
        ],
        check=True,
    )


def acquire_osf(project_id: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["osf", "-p", project_id, "clone", str(destination)],
        check=True,
    )


def finalize_raw_manifest(raw_root: Path, output: Path) -> None:
    paths = [path for path in raw_root.rglob("*") if path.is_file() and not path.name.endswith(".part")]
    entries = build_manifest(paths, raw_root)
    write_manifest(entries, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    http = subparsers.add_parser("manifest")
    http.add_argument("--manifest", type=Path, required=True)
    http.add_argument("--destination", type=Path, required=True)
    http.add_argument("--cookie")
    openneuro = subparsers.add_parser("openneuro")
    openneuro.add_argument("--dataset", required=True)
    openneuro.add_argument("--destination", type=Path, required=True)
    osf = subparsers.add_parser("osf")
    osf.add_argument("--project", required=True)
    osf.add_argument("--destination", type=Path, required=True)
    manifest = subparsers.add_parser("finalize")
    manifest.add_argument("--raw-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    http.add_argument("--minimum-free-gib", type=float, default=0)
    args = parser.parse_args()

    if args.command == "manifest":
        args.destination.mkdir(parents=True, exist_ok=True)
        completed = []
        for item in read_download_manifest(args.manifest):
            free_gib = shutil.disk_usage(args.destination).free / 2**30
            if free_gib < args.minimum_free_gib:
                raise RuntimeError(
                    f"only {free_gib:.1f} GiB free before {item.relative_path}; "
                    f"{args.minimum_free_gib:.1f} GiB required"
                )
            downloaded = resumable_http_download(item, args.destination, cookie=args.cookie)
            completed.append(
                asdict(
                    DownloadItem(
                        item.url,
                        str(downloaded),
                        item.expected_size_bytes,
                        item.sha256,
                    )
                )
            )
        print(json.dumps(completed, indent=2))
    elif args.command == "openneuro":
        acquire_openneuro(args.dataset, args.destination)
    elif args.command == "osf":
        acquire_osf(args.project, args.destination)
    else:
        finalize_raw_manifest(args.raw_root, args.output)


if __name__ == "__main__":
    main()
