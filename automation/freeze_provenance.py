"""Freeze the source/config/environment inputs used by the AutoDL queue."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDED = ("automation", "configs", "environment", "scripts", "src", "tests", "workflows")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _capture(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    output = ROOT / "results" / "provenance"
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for directory in INCLUDED:
        for path in sorted((ROOT / directory).rglob("*")):
            relative = path.relative_to(ROOT)
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and not relative.as_posix().startswith("automation/state/")
            ):
                files.append(
                    {
                        "path": relative.as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
    tree_digest = hashlib.sha256(
        "".join(f"{item['path']}\0{item['sha256']}\n" for item in files).encode()
    ).hexdigest()
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_tree_sha256": tree_digest,
        "files": files,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_head": _capture(["git", "rev-parse", "HEAD"]),
        "git_status": _capture(["git", "status", "--short"]),
        "pip_freeze_path": "environment/pip-freeze.txt",
        "environment_report_path": "environment/environment-report.json",
    }
    temporary = output / "source-freeze.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "source-freeze.json")
    print(json.dumps({"source_tree_sha256": tree_digest, "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
