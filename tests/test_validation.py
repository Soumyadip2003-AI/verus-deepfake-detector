import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from backend.app import sha256
from scripts import validate


class ValidationTests(unittest.TestCase):
    def test_calibration_never_overwrites_its_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            scores = Path(directory) / 'scores.csv'
            scores.write_text('keep me')
            with self.assertRaisesRegex(ValueError, 'paths must differ'):
                validate.calibrate(scores, Path(directory) / 'report.json', scores)
            self.assertEqual(scores.read_text(), 'keep me')

    def test_scoring_never_overwrites_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'manifest.csv'
            manifest.write_text('keep me')
            with self.assertRaisesRegex(ValueError, 'must differ'):
                validate.score_media(manifest, Path(directory), manifest)
            self.assertEqual(manifest.read_text(), 'keep me')

    def test_manifest_rejects_duplicate_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'manifest.csv'
            manifest.write_text('path,label,label,split,group\na.png,real,fake,test,a\n')
            with self.assertRaisesRegex(ValueError, 'duplicate columns'):
                validate.rows_from(manifest, {'path', 'label', 'split', 'group'})

    def test_no_face_is_reported_as_inconclusive(self):
        stats = validate.metrics([{'label': 'real', 'group': 'a', 'score': None},
                                  {'label': 'fake', 'group': 'b', 'score': .5}], .2, .8)
        self.assertEqual(stats['inconclusive'], 2)
        self.assertEqual(stats['no_face'], 1)

    def test_held_out_summary_counts_abstentions_and_auc_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, face, scores, report = (root / name for name in ('model', 'face', 'scores.csv', 'report.json'))
            model.write_bytes(b'model')
            face.write_bytes(b'face')
            hashes = {'model_sha256': sha256(model), 'face_model_sha256': sha256(face),
                      **validate.pipeline_hashes()}
            with scores.open('w', newline='') as target:
                writer = csv.DictWriter(target, fieldnames=validate.FIELDS)
                writer.writeheader()
                for label, score in [('real', .2), ('real', .5), ('real', .8), ('real', None),
                                     ('fake', .1), ('fake', .5), ('fake', .8)]:
                    writer.writerow({'path': f'{label}-{score}.png', 'label': label, 'split': 'test',
                                     'group': f'{label}-{score}', 'fake_score': '' if score is None else score,
                                     'frames_with_faces': int(score is not None), **hashes})
            with patch.object(validate, 'CHECKPOINT', model), patch.object(validate, 'FACE_MODEL', face), \
                 redirect_stdout(io.StringIO()):
                result = validate.summarize(scores, report)
            self.assertEqual(json.loads(report.read_text()), result)
            self.assertEqual(result['media'], 7)
            self.assertEqual(result['labels'], {'real': 4, 'fake': 3})
            self.assertEqual(result['no_face_rate'], 1 / 7)
            self.assertEqual(result['auroc_detectable'], 4 / 9)
            self.assertEqual((result['correct_real'], result['correct_fake']), (1, 1))
            self.assertEqual(result['accuracy_end_to_end'], 2 / 7)
            self.assertEqual(result['accuracy_when_conclusive'], 1 / 2)
            self.assertEqual(result['balanced_accuracy_end_to_end'], (1 / 4 + 1 / 3) / 2)
            self.assertEqual((result['false_real'], result['false_fake']), (1, 1))
            self.assertEqual((result['inconclusive'], result['conclusive_coverage']), (3, 4 / 7))
            with self.assertRaisesRegex(ValueError, 'paths must differ'):
                validate.summarize(scores, scores)

    def test_calibration_uses_separate_source_groups_and_held_out_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, face = root / 'model', root / 'face'
            model.write_bytes(b'model')
            face.write_bytes(b'face')
            scores, report, thresholds = root / 'scores.csv', root / 'report.json', root / 'thresholds.json'
            with scores.open('w', newline='') as target:
                writer = csv.DictWriter(target, fieldnames=validate.FIELDS)
                writer.writeheader()
                for split in ('calibration', 'test'):
                    for label in ('real', 'fake'):
                        for index in range(100):
                            writer.writerow({'path': f'{split}-{label}-{index}.png', 'label': label,
                                             'split': split, 'group': f'{split}-{label}-{index}',
                                             'slice': 'image', 'method': 'Deepfakes' if label == 'fake' else 'original',
                                             'fake_score': .1 if label == 'real' else .9,
                                             'frames_with_faces': 1, 'model_sha256': sha256(model),
                                             'face_model_sha256': sha256(face), **validate.pipeline_hashes()})
            with patch.object(validate, 'CHECKPOINT', model), patch.object(validate, 'FACE_MODEL', face), \
                 redirect_stdout(io.StringIO()):
                self.assertTrue(validate.calibrate(scores, report, thresholds))
            self.assertTrue(json.loads(report.read_text())['passed'])
            self.assertEqual(json.loads(report.read_text())['test_methods']['Deepfakes']['false_real_source_groups'], 0)
            self.assertEqual(json.loads(thresholds.read_text())['real_max'], .1)
            self.assertEqual(json.loads(thresholds.read_text())['fake_min'], .9)
            self.assertEqual(json.loads(thresholds.read_text())['version'], validate.THRESHOLD_VERSION)
            self.assertEqual(json.loads(thresholds.read_text())['pipeline_sha256'],
                             validate.pipeline_hashes()['pipeline_sha256'])

            with scores.open(newline='') as source:
                rows = list(csv.DictReader(source))
            for row in rows:
                if row['split'] == 'test':
                    row['fake_score'] = .9 if row['label'] == 'real' else .1
            with scores.open('w', newline='') as target:
                writer = csv.DictWriter(target, fieldnames=validate.FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with patch.object(validate, 'CHECKPOINT', model), patch.object(validate, 'FACE_MODEL', face), \
                 redirect_stdout(io.StringIO()):
                self.assertFalse(validate.calibrate(scores, report, thresholds))
            self.assertFalse(thresholds.exists())
            self.assertFalse(json.loads(report.read_text())['passed'])

    def test_scored_rows_rejects_stale_inference_and_inconsistent_face_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, face, scores = root / 'model', root / 'face', root / 'scores.csv'
            model.write_bytes(b'model')
            face.write_bytes(b'face')
            row = {'path': 'sample.png', 'label': 'real', 'split': 'calibration', 'group': 'sample',
                   'fake_score': .2, 'frames_with_faces': 1, 'model_sha256': sha256(model),
                   'face_model_sha256': sha256(face), **validate.pipeline_hashes()}

            def write_row():
                with scores.open('w', newline='') as target:
                    writer = csv.DictWriter(target, fieldnames=validate.FIELDS)
                    writer.writeheader()
                    writer.writerow(row)

            with patch.object(validate, 'CHECKPOINT', model), patch.object(validate, 'FACE_MODEL', face):
                row['pipeline_sha256'] = 'stale'
                write_row()
                with self.assertRaisesRegex(ValueError, 'different model files or inference code'):
                    validate.scored_rows(scores)
                row['pipeline_sha256'] = validate.pipeline_hashes()['pipeline_sha256']
                row['frames_with_faces'] = 0
                write_row()
                with self.assertRaisesRegex(ValueError, 'Score and face count disagree'):
                    validate.scored_rows(scores)
                report, thresholds = root / 'report.json', root / 'thresholds.json'
                report.write_text('stale')
                thresholds.write_text('stale')
                with self.assertRaisesRegex(ValueError, 'Score and face count disagree'):
                    validate.calibrate(scores, report, thresholds)
                self.assertFalse(report.exists())
                self.assertFalse(thresholds.exists())

    def test_no_face_media_counts_against_end_to_end_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, face = root / 'model', root / 'face'
            model.write_bytes(b'model')
            face.write_bytes(b'face')
            scores, report, thresholds = root / 'scores.csv', root / 'report.json', root / 'thresholds.json'
            hashes = {'model_sha256': sha256(model), 'face_model_sha256': sha256(face),
                      **validate.pipeline_hashes()}
            with scores.open('w', newline='') as target:
                writer = csv.DictWriter(target, fieldnames=validate.FIELDS)
                writer.writeheader()
                for split in ('calibration', 'test'):
                    for label in ('real', 'fake'):
                        for index in range(11):
                            detected = index < 10
                            writer.writerow({'path': f'{split}-{label}-{index}.png', 'label': label,
                                             'split': split, 'group': f'{split}-{label}-{index}',
                                             'method': 'Deepfakes' if label == 'fake' else 'original',
                                             'fake_score': (.1 if label == 'real' else .9) if index < 2 else
                                                           (.5 if detected else ''),
                                             'frames_with_faces': int(detected), **hashes})
            with patch.object(validate, 'CHECKPOINT', model), patch.object(validate, 'FACE_MODEL', face), \
                 patch.object(validate, 'MIN_PER_CLASS', 10), patch.object(validate, 'MIN_GROUPS_PER_CLASS', 10), \
                 patch.object(validate, 'MAX_ERROR_UPPER', .5), redirect_stdout(io.StringIO()):
                self.assertFalse(validate.calibrate(scores, report, thresholds))
            self.assertIn('No thresholds meet', json.loads(report.read_text())['problems'][0])
            self.assertFalse(thresholds.exists())

    def test_scoring_rejects_duplicate_media_behind_distinct_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / 'sample.png'
            media.touch()
            manifest, output = root / 'manifest.csv', root / 'scores.csv'
            with manifest.open('w', newline='') as target:
                writer = csv.DictWriter(target, fieldnames=('path', 'label', 'split', 'group'))
                writer.writeheader()
                writer.writerow({'path': 'sample.png', 'label': 'real', 'split': 'calibration', 'group': 'a'})
                writer.writerow({'path': 'sub/../sample.png', 'label': 'fake', 'split': 'test', 'group': 'b'})
            (root / 'sub').mkdir()
            output.write_text('stale')

            class StubDetector:
                def analyze(self, path, is_video):
                    return {'fake_score': .2, 'frames_with_faces': 1}

            with patch.object(validate, 'CHECKPOINT', media), patch.object(validate, 'FACE_MODEL', media), \
                 patch.object(validate, 'Detector', StubDetector):
                with self.assertRaisesRegex(ValueError, 'Duplicate media file'):
                    validate.score_media(manifest, root, output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
