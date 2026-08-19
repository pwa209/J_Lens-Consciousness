#!/usr/bin/env python3
"""Audit precomputed NRP remote indexes against manifest and completed local archives."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from remote_tar_extract import SelectedMember, index_local, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--indexes", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if Path(row["relative_path"]).name.endswith("_NRP.tar")
        ]
    audited = []
    exact_local_comparisons = 0
    for row in rows:
        archive_name = Path(row["relative_path"]).name
        name = Path(archive_name).stem
        index_path = args.indexes / f"{name}.index.json"
        if not index_path.exists():
            raise RuntimeError(f"missing sparse index: {index_path}")
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        if metadata["requested_url"] != row["url"]:
            raise RuntimeError(f"URL mismatch in {index_path}")
        if int(metadata["size_bytes"]) != int(row["expected_size_bytes"]):
            raise RuntimeError(f"size mismatch in {index_path}")
        if payload.get("index_mode") != "sparse_subtree_skip":
            raise RuntimeError(f"index was not produced by sparse mode: {index_path}")
        indexed = [SelectedMember(**member) for member in payload["selected"]]
        archive = args.archive_root / archive_name
        local_exact = False
        if archive.exists():
            local, _ = index_local(archive)
            if indexed != local:
                indexed_by_path = {member.path: member for member in indexed}
                local_by_path = {member.path: member for member in local}
                diagnostic = {
                    "archive": str(archive),
                    "indexed_count": len(indexed),
                    "local_count": len(local),
                    "missing_from_index": sorted(local_by_path.keys() - indexed_by_path.keys()),
                    "extra_in_index": sorted(indexed_by_path.keys() - local_by_path.keys()),
                    "changed": [
                        {
                            "path": path,
                            "indexed": indexed_by_path[path].__dict__,
                            "local": local_by_path[path].__dict__,
                        }
                        for path in sorted(indexed_by_path.keys() & local_by_path.keys())
                        if indexed_by_path[path] != local_by_path[path]
                    ],
                }
                raise RuntimeError(
                    "index differs from completed local archive: "
                    + json.dumps(diagnostic, sort_keys=True)
                )
            local_exact = True
            exact_local_comparisons += 1
        audited.append(
            {
                "name": name,
                "index": str(index_path),
                "index_sha256": sha256_file(index_path),
                "archive_bytes": int(metadata["size_bytes"]),
                "selected_bytes": sum(member.size_bytes for member in indexed),
                "selected_count": len(indexed),
                "exact_local_archive_comparison": local_exact,
            }
        )
    payload = {
        "status": "PASS",
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "index_count": len(audited),
        "exact_local_archive_comparisons": exact_local_comparisons,
        "archive_bytes": sum(item["archive_bytes"] for item in audited),
        "selected_bytes": sum(item["selected_bytes"] for item in audited),
        "avoided_bytes": sum(
            item["archive_bytes"] - item["selected_bytes"] for item in audited
        ),
        "indexes": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
