"""Create a content-blind, hash-stamped inventory of pre-Stage-1 human outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusions(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-root", type=Path, default=Path("results/human"))
    parser.add_argument(
        "--firewall", type=Path, default=Path("configs/execution/pilot_firewall.tsv")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/registration/pilot-exposure-inventory.json"),
    )
    args = parser.parse_args()
    human_root = args.human_root if args.human_root.is_absolute() else ROOT / args.human_root
    firewall = args.firewall if args.firewall.is_absolute() else ROOT / args.firewall
    records: list[dict[str, object]] = []
    for row in exclusions(firewall):
        directory = human_root / row["dataset_id"] / row["participant_id"]
        for path in sorted(candidate for candidate in directory.rglob("*") if candidate.is_file()):
            stat = path.stat()
            records.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": sha256(path),
                }
            )
    payload = {
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "content_blind_inventory": True,
        "files": records,
        "file_count": len(records),
        "declared_inspection": [
            "Gabor sub-10 preprocessing QC",
            "Gabor sub-10 fold-3 model QC",
        ],
        "prohibited_until_ipa": [
            "condition contrasts",
            "effect sizes",
            "group summaries",
            "human-machine architecture rankings",
        ],
        "confirmatory_exclusions": exclusions(firewall),
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "file_count": len(records)}, indent=2))


if __name__ == "__main__":
    main()

