#!/usr/bin/env python3
"""Print parsed POSIX tar headers from a known aligned byte offset."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from remote_tar_extract import _next_tar_offset, _parse_header


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    offset = args.block * tarfile.BLOCKSIZE
    root = None
    with args.archive.open("rb") as handle:
        for _ in range(args.count):
            handle.seek(offset)
            raw = handle.read(tarfile.BLOCKSIZE)
            member = _parse_header(raw, offset, root)
            if member is None:
                print(f"INVALID offset={offset} block={offset // tarfile.BLOCKSIZE} raw={raw[:16]!r}")
                break
            if root is None:
                root = member.name.rstrip("/").split("/", 1)[0]
            print(
                f"block={offset // tarfile.BLOCKSIZE} type={member.type!r} "
                f"size={member.size} name={member.name}"
            )
            offset = _next_tar_offset(member)


if __name__ == "__main__":
    main()
