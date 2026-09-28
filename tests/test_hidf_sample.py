import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path
from unittest.mock import patch

from scripts import hidf_sample


class HiDFSampleTests(unittest.TestCase):
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
