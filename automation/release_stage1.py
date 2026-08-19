"""Record a verifiable in-principle acceptance before human confirmatory work."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipa-url", required=True)
    parser.add_argument("--venue", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("results/registration/STAGE1_IPA.json")
    )
    args = parser.parse_args()
    parsed = urlparse(args.ipa_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise SystemExit("--ipa-url must be a public HTTPS Stage 1/IPA record")
    payload = {
        "in_principle_acceptance": True,
        "venue": args.venue,
        "url": args.ipa_url,
        "received_at_utc": datetime.now(UTC).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

