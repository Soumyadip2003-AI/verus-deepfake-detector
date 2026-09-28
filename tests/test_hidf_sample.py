import tempfile
import unittest
import zipfile
import zlib
import random
from pathlib import Path
from unittest.mock import patch

from scripts import hidf_sample


class HiDFSampleTests(unittest.TestCase):
    def test_range_download_retries_an_oversized_response(self):
        class Response:
            status = 206
            headers = {"Content-Range": "bytes 0-2/10"}

            def __init__(self, data):
                self.data = data

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def read(self, _):
                return self.data

        responses = [Response(b"four"), Response(b"abc")]
        with patch.object(hidf_sample.urllib.request, "urlopen", side_effect=responses) as urlopen, \
             patch.object(hidf_sample.time, "sleep"):
            self.assertEqual(hidf_sample.get("https://example.invalid/archive", 0, 2), b"abc")
        self.assertEqual(urlopen.call_count, 2)

    def test_split_keeps_reused_target_identity_together(self):
        def member(name):
            return hidf_sample.Member(name, 0, 0, 0, 0, 0, 0)

        fake = {
            "c00001": member("Fake-vid/c00001_f00001.mp4"),
            "c00002": member("Fake-vid/c00002_f00001.mp4"),
            "c00003": member("Fake-vid/c00003_f00003.mp4"),
            "c00004": member("Fake-vid/c00004_f00004.mp4"),
        }
        splits, groups = hidf_sample.assign_splits(list(fake), fake, 2, random.Random(7))
        self.assertEqual(splits["c00001"], splits["c00002"])
        self.assertEqual(groups["c00001"], groups["c00002"])
        self.assertEqual(sum(split == "calibration" for split in splits.values()), 2)

    def test_extract_verifies_and_writes_one_zip_member(self):
        payload = b"video bytes" * 100
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "sample.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as target:
                target.writestr("Real-vid/c00001.mp4", payload)
            raw = archive.read_bytes()
            with zipfile.ZipFile(archive) as source:
                info = source.getinfo("Real-vid/c00001.mp4")
                central = raw.index(b"PK\x01\x02")
            member = hidf_sample.Member(info.filename, info.header_offset, central - 1,
                                        info.compress_size, info.file_size, info.CRC, info.compress_type)
            with patch.object(hidf_sample, "get", return_value=raw[info.header_offset:central]):
                hidf_sample.extract("Real-vid.zip", member, root / "output")
            extracted = root / "output" / info.filename
            self.assertEqual(extracted.read_bytes(), payload)
            self.assertEqual(zlib.crc32(extracted.read_bytes()), info.CRC)


if __name__ == "__main__":
    unittest.main()
