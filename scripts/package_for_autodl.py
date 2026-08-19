"""Create a source-only upload archive with a SHA-256 sidecar."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
OUTPUT = DIST / "jacobian-conscious-access-upload.zip"
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".snakemake",
    "dist",
}
EXCLUDED_PREFIXES = {
    ("data", "raw"),
    ("data", "derivatives"),
    ("results",),
}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    return not any(relative.parts[: len(prefix)] == prefix for prefix in EXCLUDED_PREFIXES)


DIST.mkdir(exist_ok=True)
with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and included(path):
            archive.write(path, path.relative_to(ROOT).as_posix())

digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
OUTPUT.with_suffix(".zip.sha256").write_text(
    f"{digest}  {OUTPUT.name}\n",
    encoding="utf-8",
)
print(OUTPUT)
print(digest)

