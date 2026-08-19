from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from jacaccess.io.download import DownloadItem
from jacaccess.io.ranged_download import ranged_download, split_ranges
from jacaccess.machine.accuracy_matching import (
    ARCHITECTURES,
    stabilized_ipw_table,
    summarize_accuracy_match,
)


class _RangeHandler(BaseHTTPRequestHandler):
    data = bytes(range(251)) * 4096
    etag = '"range-test-v1"'

    def _headers(self, status: int, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", self.etag)
        self.end_headers()

    def do_HEAD(self) -> None:
        self._headers(200, len(self.data))

    def do_GET(self) -> None:
        value = self.headers.get("Range")
        if not value or not value.startswith("bytes="):
            self._headers(200, len(self.data))
            self.wfile.write(self.data)
            return
        start_text, end_text = value.removeprefix("bytes=").split("-", maxsplit=1)
        start, end = int(start_text), int(end_text)
        payload = self.data[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(self.data)}")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", self.etag)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class RangedDownloadTests(unittest.TestCase):
    def test_ranges_cover_interval_once(self) -> None:
        ranges = split_ranges(7, 101, 8)
        flattened = [value for start, end in ranges for value in range(start, end + 1)]
        self.assertEqual(flattened, list(range(7, 101)))

    def test_parallel_download_adopts_sequential_prefix(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                destination = root / "artifact.bin"
                prefix = destination.with_suffix(".bin.part")
                prefix.write_bytes(_RangeHandler.data[:12345])
                item = DownloadItem(
                    url=f"http://127.0.0.1:{server.server_port}/artifact.bin",
                    relative_path="artifact.bin",
                    expected_size_bytes=len(_RangeHandler.data),
                    sha256=hashlib.sha256(_RangeHandler.data).hexdigest(),
                )
                completed = ranged_download(item, root, connections=4, chunk_bytes=1024)
                self.assertEqual(completed.read_bytes(), _RangeHandler.data)
                self.assertFalse(prefix.exists())
                self.assertFalse(destination.with_name("artifact.bin.ranges").exists())
                receipt = json.loads(
                    destination.with_name("artifact.bin.download.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(receipt["status"], "PASS")
                self.assertEqual(receipt["adopted_prefix_bytes"], 12345)
                self.assertEqual(ranged_download(item, root), destination)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


class AccuracyMatchingTests(unittest.TestCase):
    @staticmethod
    def _table(matched: bool = True) -> pd.DataFrame:
        rows = []
        bin_zero = (69, 70, 71, 70, 70) if matched else (50, 60, 70, 80, 90)
        for seed in range(2):
            for architecture, correct in zip(ARCHITECTURES, bin_zero, strict=True):
                for difficulty_bin, value in ((0, correct), (1, 50 + 10 * ARCHITECTURES.index(architecture))):
                    rows.append(
                        {
                            "architecture": architecture,
                            "seed": seed,
                            "split": "validation",
                            "difficulty_bin": difficulty_bin,
                            "correct_count": value,
                            "sample_count": 100,
                            "presence_accuracy": value / 100,
                        }
                    )
        return pd.DataFrame(rows)

    def test_selects_matched_bin_closest_to_point_seven(self) -> None:
        summary, selected = summarize_accuracy_match(
            self._table(), tolerance=0.02, target_accuracy=0.70
        )
        self.assertEqual(selected, 0)
        self.assertTrue(bool(summary.loc[summary["difficulty_bin"] == 0, "within_tolerance"].iloc[0]))
        weights = stabilized_ipw_table(self._table())
        self.assertTrue((weights[["weight_if_correct", "weight_if_incorrect"]] > 0).all().all())

    def test_uses_weighted_fallback_when_no_bin_matches(self) -> None:
        table = self._table(matched=False)
        table.loc[table["difficulty_bin"] == 1, "correct_count"] = table.loc[
            table["difficulty_bin"] == 1, "architecture"
        ].map(dict(zip(ARCHITECTURES, (40, 52, 64, 76, 88), strict=True)))
        table["presence_accuracy"] = table["correct_count"] / table["sample_count"]
        _summary, selected = summarize_accuracy_match(
            table, tolerance=0.02, target_accuracy=0.70
        )
        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
