import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.faceforensics_manifest import build_manifest


class FaceForensicsManifestTests(unittest.TestCase):
    def test_rejects_duplicate_method_before_writing_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'manifest.csv'
            with self.assertRaisesRegex(ValueError, 'distinct'):
                build_manifest(Path(directory), Path(directory), ('Deepfakes', 'Deepfakes'), output)
            self.assertFalse(output.exists())

    def test_official_pairs_keep_original_and_fake_by_target_source(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, splits, output = base / 'dataset', base / 'splits', base / 'manifest.csv'
            splits.mkdir()
            (splits / 'val.json').write_text(json.dumps([['001', '002']]))
            (splits / 'test.json').write_text(json.dumps([['101', '102']]))
            (splits / 'train.json').write_text(json.dumps([['201', '202']]))
            for target, source in (('001', '002'), ('002', '001'), ('101', '102'), ('102', '101')):
                for relative in (f'original_sequences/youtube/c23/videos/{target}.mp4',
                                 f'manipulated_sequences/Deepfakes/c23/videos/{target}_{source}.mp4'):
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
            self.assertEqual(build_manifest(root, splits, ('Deepfakes',), output), 8)
            with output.open(newline='') as source:
                rows = list(csv.DictReader(source))
            self.assertEqual({row['group'] for row in rows if row['split'] == 'calibration'}, {'001', '002'})
            self.assertEqual({row['group'] for row in rows if row['split'] == 'test'}, {'101', '102'})
            self.assertEqual({row['label'] for row in rows}, {'real', 'fake'})
            self.assertEqual({row['method'] for row in rows if row['label'] == 'fake'}, {'Deepfakes'})


if __name__ == '__main__':
    unittest.main()
