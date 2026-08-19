from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from automation.acquire_kronemer_hybrid import schedule_items
from jacaccess.io.download import DownloadItem


class HybridScheduleTests(unittest.TestCase):
    def test_interleaves_network_and_local_without_losing_items(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            downloads = root / "downloads"
            raw = root / "raw"
            inventories = root / "inventories"
            items = [
                DownloadItem("https://example.test/a", "kronemer/a.tar", 1),
                DownloadItem("https://example.test/b", "kronemer/b.tar", 1),
                DownloadItem("https://example.test/c", "kronemer/c.tar", 1),
                DownloadItem("https://example.test/d", "kronemer/d.tar", 1),
            ]

            (raw / "a").mkdir(parents=True)
            (raw / "a" / ".full_extraction_complete").touch()
            inventories.mkdir(parents=True)
            (inventories / "a.json").write_text("{}\n", encoding="utf-8")
            (downloads / "kronemer").mkdir(parents=True)
            (downloads / "kronemer" / "b.tar").touch()

            scheduled = schedule_items(items, downloads, raw, inventories)

            self.assertEqual(
                [Path(item.relative_path).stem for item in scheduled],
                ["a", "c", "b", "d"],
            )
            self.assertEqual(set(scheduled), set(items))


if __name__ == "__main__":
    unittest.main()
