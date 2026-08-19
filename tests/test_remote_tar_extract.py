from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from automation.remote_tar_extract import (
    HTTPRangeReader,
    RemoteMetadata,
    index_local,
    index_remote_sparse,
    selected_from_tar,
)


class FakeResponse:
    def __init__(self, payload: bytes, start: int, end: int) -> None:
        self.status_code = 206
        self.content = payload[start : end + 1]
        self.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}

    def raise_for_status(self) -> None:
        return None


class FakeSession:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def get(self, _url: str, *, headers: dict[str, str], timeout: float) -> FakeResponse:
        del timeout
        start, end = [int(value) for value in headers["Range"][6:].split("-")]
        return FakeResponse(self.payload, start, end)


def make_tar() -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        files = {
            "subject/EEG_Session/EEG_Data/run.raw": b"eeg" * 10000,
            "subject/EEG_Session/Behavioral_Data/events.csv": b"trial,outcome\n1,seen\n",
            "subject/MRI_Session/MRI_Data/image.dcm": b"mri" * 1000,
            "subject/video/movie.mp4": b"video" * 1000,
        }
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def make_multimodal_tar() -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.GNU_FORMAT) as archive:
        entries = [
            ("subject/Center/EEG_Session/EEG_Data/run.raw", b"eeg" * 1000),
            ("subject/Center/MRI_Session/Behavioral_Data/events.csv", b"trial,seen\n"),
            (
                "subject/Center/MRI_Session/Behavioral_Data/"
                + "very-long-session-name-" * 5
                + ".csv",
                b"trial,seen\n",
            ),
            ("subject/Center/MRI_Session/MRI_Data/", None),
            ("subject/Center/MRI_Session/MRI_Data/sidecar.json", b"{}"),
            *[
                (
                    f"subject/Center/MRI_Session/MRI_Data/image-{index:02d}.dcm",
                    bytes([index]) * (2 * 1024 * 1024),
                )
                for index in range(20)
            ],
            ("subject/Quadrant/", None),
            ("subject/Quadrant/EEG_Session/EEG_Data/run.raw", b"eeg2" * 1000),
            ("subject/Quadrant/MRI_Session/MRI_Data/", None),
            ("subject/Quadrant/MRI_Session/MRI_Data/c.dcm", b"c" * (2 * 1024 * 1024)),
        ]
        for name, payload in entries:
            info = tarfile.TarInfo(name)
            if payload is None:
                info.type = tarfile.DIRTYPE
                info.size = 0
                archive.addfile(info)
            else:
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


class RemoteTarTests(unittest.TestCase):
    def test_seekable_http_reader_indexes_same_selected_members(self) -> None:
        payload = make_tar()
        metadata = RemoteMetadata(
            requested_url="https://example.test/archive.tar",
            final_url="https://example.test/archive.tar",
            size_bytes=len(payload),
            etag=None,
            last_modified=None,
            accepts_ranges=True,
        )
        reader = HTTPRangeReader(
            metadata,
            block_bytes=4096,
            session=FakeSession(payload),
        )
        with tarfile.open(fileobj=reader, mode="r:") as archive:
            remote_selected, remote_skipped = selected_from_tar(archive)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "archive.tar"
            path.write_bytes(payload)
            local_selected, local_skipped = index_local(path)
        self.assertEqual(remote_selected, local_selected)
        self.assertEqual(remote_skipped, local_skipped)
        self.assertEqual(
            [member.path for member in remote_selected],
            [
                "subject/EEG_Session/EEG_Data/run.raw",
                "subject/EEG_Session/Behavioral_Data/events.csv",
            ],
        )

    def test_reader_seek_and_cache(self) -> None:
        payload = bytes(range(256)) * 100
        metadata = RemoteMetadata(
            requested_url="https://example.test/archive.tar",
            final_url="https://example.test/archive.tar",
            size_bytes=len(payload),
            etag=None,
            last_modified=None,
            accepts_ranges=True,
        )
        reader = HTTPRangeReader(
            metadata,
            block_bytes=1024,
            session=FakeSession(payload),
        )
        reader.seek(900)
        self.assertEqual(reader.read(100), payload[900:1000])
        first_requests = reader.request_count
        reader.seek(950)
        self.assertEqual(reader.read(50), payload[950:1000])
        self.assertEqual(reader.request_count, first_requests)

    def test_sparse_index_skips_mri_data_and_matches_local_selection(self) -> None:
        payload = make_multimodal_tar()
        metadata = RemoteMetadata(
            requested_url="https://example.test/archive.tar",
            final_url="https://example.test/archive.tar",
            size_bytes=len(payload),
            etag=None,
            last_modified=None,
            accepts_ranges=True,
        )
        original_session = __import__("requests").Session
        try:
            __import__("requests").Session = lambda: FakeSession(payload)
            sparse, statistics = index_remote_sparse(
                metadata,
                timeout=1,
                retries=1,
                probe_bytes=4 * 1024 * 1024,
            )
        finally:
            __import__("requests").Session = original_session
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "archive.tar"
            path.write_bytes(payload)
            local, _ = index_local(path)
        self.assertEqual(sparse, local)
        self.assertEqual(statistics["skipped_subtree_count"], 2)
        self.assertLess(statistics["index_transferred_bytes"], len(payload))


if __name__ == "__main__":
    unittest.main()
