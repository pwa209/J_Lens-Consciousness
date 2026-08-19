"""Build the scalp-EEG Kronemer manifest from the public NITRC file list."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import requests

FILE_RE = re.compile(r"^\d+_(?:RP_EEG|NRP)\.tar$")


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.href: str | None = None
        self.text: list[str] = []
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        mapping = dict(attrs)
        self.href = mapping.get("href")
        self.text = []

    def handle_data(self, data: str) -> None:
        if self.href is not None:
            self.text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self.href is not None:
            self.links.append((self.href, "".join(self.text).strip()))
            self.href = None
            self.text = []


def _included_kronemer_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["participant_id"].strip()
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("dataset_id") == "kronemer"
            and str(row.get("include", "")).strip().lower() in {"1", "true", "yes"}
        }


def _participant_selected(filename: str, participant_ids: set[str]) -> bool:
    stem = Path(filename).stem
    numeric = stem.partition("_")[0]
    return stem in participant_ids or numeric in participant_ids


def _resolve_size(item: tuple[str, str]) -> tuple[str, str, int]:
    name, link = item
    head = requests.head(link, allow_redirects=True, timeout=30)
    head.raise_for_status()
    size = head.headers.get("Content-Length")
    if size is None:
        raise RuntimeError(f"no Content-Length for {name}")
    return name, link, int(size)


def build(
    output: Path,
    url: str,
    *,
    participant_ids: set[str] | None = None,
    workers: int = 16,
    skip_sizes: bool = False,
) -> int:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    parser = LinkParser()
    parser.feed(response.text)
    files: dict[str, str] = {}
    for href, text in parser.links:
        name = Path(text).name
        if FILE_RE.fullmatch(name) and "download" in href:
            files[name] = urljoin(url, href)
    if not files:
        raise RuntimeError("NITRC file list yielded no RP_EEG or NRP archives")
    if participant_ids is not None:
        files = {
            name: link
            for name, link in files.items()
            if _participant_selected(name, participant_ids)
        }
        if not files:
            raise RuntimeError("none of the included Kronemer participants match NITRC archives")

    rows: list[dict[str, str | int]] = []
    if skip_sizes:
        resolved = [(name, files[name], "") for name in sorted(files)]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            resolved = sorted(executor.map(_resolve_size, files.items()))
    for name, link, size in resolved:
        rows.append(
            {
                "url": link,
                "relative_path": f"kronemer/{name}",
                "expected_size_bytes": size,
                "sha256": "",
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("url", "relative_path", "expected_size_bytes", "sha256"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", default="https://www.nitrc.org/frs/?group_id=1550"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/data_sources/kronemer_downloads.tsv"),
    )
    parser.add_argument("--participants", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip-sizes", action="store_true")
    args = parser.parse_args()
    participant_ids = (
        _included_kronemer_ids(args.participants) if args.participants is not None else None
    )
    count = build(
        args.output,
        args.url,
        participant_ids=participant_ids,
        workers=args.workers,
        skip_sizes=args.skip_sizes,
    )
    print(f"wrote {count} Kronemer scalp-EEG archives")


if __name__ == "__main__":
    main()
