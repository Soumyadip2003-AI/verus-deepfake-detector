import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import backend.app as backend


class VideoFrameTests(unittest.TestCase):
    def test_video_frames_are_decoded_sequentially_and_bounded(self):
        class Capture:
            def __init__(self):
                self.index = 0

            def get(self, _):
                return 1000

            def read(self):
                frame = np.full((1, 1, 3), self.index, dtype=np.uint8)
                self.index += 1
                return True, frame

        capture = Capture()
        frames = backend.read_video_frames(capture)
        self.assertEqual(capture.index, backend.MAX_VIDEO_SCAN_FRAMES)
        self.assertEqual([int(frame[0, 0, 0]) for frame in frames],
                         np.linspace(0, 239, backend.MAX_FRAMES, dtype=int).tolist())


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.threshold_patch = patch.object(backend, 'THRESHOLDS_FILE', Path(self.temp.name) / 'missing.json')
        self.production_patch = patch.object(backend, 'PRODUCTION', False)
        self.threshold_patch.start()
        self.production_patch.start()
        backend.thresholds.cache_clear()
        self.client = TestClient(backend.app)
        self.original_detector = backend._detector

    def tearDown(self):
        backend._detector = self.original_detector
        backend.thresholds.cache_clear()
        self.production_patch.stop()
        self.threshold_patch.stop()
        self.temp.cleanup()

    def test_rejects_unsupported_file(self):
        response = self.client.post('/api/analyze', files={'file': ('note.txt', b'hello', 'text/plain')})
        self.assertEqual(response.status_code, 415)

    def test_valid_upload_reaches_detector_and_removes_temporary_file(self):
        image = io.BytesIO()
        Image.new('RGB', (16, 16), 'white').save(image, format='PNG')
        seen = []

        class StubDetector:
            def analyze(self, path, is_video):
                seen.append(Path(path))
                assert path.is_file() and not is_video
                return {'verdict': 'inconclusive', 'fake_score': None, 'frames_sampled': 1,
                        'frames_with_faces': 0, 'multiple_faces': False, 'model': 'test'}

        backend._detector = StubDetector()
        response = self.client.post('/api/analyze', files={'file': ('face.png', image.getvalue(), 'image/png')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['verdict'], 'inconclusive')
        self.assertFalse(seen[0].exists())

    def test_production_refuses_uncalibrated_results(self):
        with TemporaryDirectory() as temp, patch.object(backend, 'THRESHOLDS_FILE', Path(temp) / 'missing.json'), \
             patch.object(backend, 'PRODUCTION', True):
            backend.thresholds.cache_clear()
            self.assertFalse(self.client.get('/api/health').json()['ready'])
            response = self.client.post('/api/analyze', files={'file': ('face.png', b'png', 'image/png')})
            self.assertEqual(response.status_code, 503)
            backend.thresholds.cache_clear()

    def test_classification_abstains_between_thresholds(self):
        self.assertEqual(backend.classify(None, .2, .8), 'inconclusive')
        self.assertEqual(backend.classify(.1, .2, .8), 'no_strong_signal')
        self.assertEqual(backend.classify(.5, .2, .8), 'inconclusive')
        self.assertEqual(backend.classify(.9, .2, .8), 'likely_manipulated')

    def test_production_rejects_thresholds_for_another_model(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint, face, config = root / 'model', root / 'face', root / 'thresholds.json'
            checkpoint.write_bytes(b'model')
            face.write_bytes(b'face')
            config.write_text(json.dumps({'version': backend.THRESHOLD_VERSION, 'real_max': .2, 'fake_min': .8,
                                          'model_sha256': 'wrong', 'face_model_sha256': backend.sha256(face)}))
            with patch.object(backend, 'CHECKPOINT', checkpoint), patch.object(backend, 'FACE_MODEL', face), \
                 patch.object(backend, 'THRESHOLDS_FILE', config), patch.object(backend, 'PRODUCTION', True):
                backend.thresholds.cache_clear()
                self.assertFalse(self.client.get('/api/health').json()['ready'])
                backend.thresholds.cache_clear()

    def test_health_and_analysis_reject_a_broken_detector(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint, face = root / 'model', root / 'face'
            checkpoint.write_bytes(b'corrupt')
            face.write_bytes(b'face')
            with patch.object(backend, 'CHECKPOINT', checkpoint), patch.object(backend, 'FACE_MODEL', face), \
                 patch.object(backend, 'THRESHOLDS_FILE', root / 'missing.json'), \
                 patch.object(backend, 'PRODUCTION', False), patch.object(backend, '_detector', None), \
                 patch.object(backend, 'Detector', side_effect=RuntimeError('bad weights')), \
                 patch.object(backend.logging, 'exception'):
                backend.thresholds.cache_clear()
                health = self.client.get('/api/health').json()
                self.assertFalse(health['ready'])
                self.assertIn('could not load', health['problem'])
                response = self.client.post('/api/analyze', files={'file': ('face.png', b'png', 'image/png')})
                self.assertEqual(response.status_code, 503)
                backend.thresholds.cache_clear()

    def test_thresholds_reject_changed_inference_pipeline(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint, face, config = root / 'model', root / 'face', root / 'thresholds.json'
            checkpoint.write_bytes(b'model')
            face.write_bytes(b'face')
            config.write_text(json.dumps({'version': backend.THRESHOLD_VERSION, 'real_max': .2, 'fake_min': .8,
                                          'model_sha256': backend.sha256(checkpoint),
                                          'face_model_sha256': backend.sha256(face),
                                          **backend.pipeline_hashes(), 'pipeline_sha256': 'stale'}))
            with patch.object(backend, 'CHECKPOINT', checkpoint), patch.object(backend, 'FACE_MODEL', face), \
                 patch.object(backend, 'THRESHOLDS_FILE', config):
                backend.thresholds.cache_clear()
                with self.assertRaisesRegex(ValueError, 'different inference pipeline'):
                    backend.thresholds()
                backend.thresholds.cache_clear()

    def test_thresholds_reject_previous_validation_rules(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint, face, config = root / 'model', root / 'face', root / 'thresholds.json'
            checkpoint.write_bytes(b'model')
            face.write_bytes(b'face')
            config.write_text(json.dumps({'version': 1, 'real_max': .2, 'fake_min': .8,
                                          'model_sha256': backend.sha256(checkpoint),
                                          'face_model_sha256': backend.sha256(face), **backend.pipeline_hashes()}))
            with patch.object(backend, 'CHECKPOINT', checkpoint), patch.object(backend, 'FACE_MODEL', face), \
                 patch.object(backend, 'THRESHOLDS_FILE', config):
                backend.thresholds.cache_clear()
                with self.assertRaisesRegex(ValueError, 'configuration is invalid'):
                    backend.thresholds()
                backend.thresholds.cache_clear()

    def test_malformed_thresholds_report_unready_instead_of_crashing(self):
        with TemporaryDirectory() as temp:
            config = Path(temp) / 'thresholds.json'
            config.write_text('{"real_max": null, "fake_min": 0.8}')
            with patch.object(backend, 'THRESHOLDS_FILE', config):
                backend.thresholds.cache_clear()
                health = self.client.get('/api/health')
                self.assertEqual(health.status_code, 200)
                self.assertFalse(health.json()['ready'])
                response = self.client.post('/api/analyze', files={'file': ('face.png', b'png', 'image/png')})
                self.assertEqual(response.status_code, 503)
                backend.thresholds.cache_clear()


if __name__ == '__main__':
    unittest.main()
