"""Local deepfake detection API and frontend server."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from safetensors.torch import load_model
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

from backend.photo_detector import (CHECKPOINT as PHOTO_CHECKPOINT, PhotoDetector,
                                    pipeline_hashes as photo_pipeline_hashes)

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
MODELS = ROOT / "backend" / "models"
CHECKPOINT = MODELS / "gend-clip-l14.safetensors"
FACE_MODEL = MODELS / "face_detection_yunet_2023mar.onnx"
THRESHOLDS_FILE = Path(os.environ.get("VERITY_THRESHOLDS_FILE", str(MODELS / "thresholds.json")))
PHOTO_THRESHOLDS_FILE = Path(os.environ.get("VERITY_PHOTO_THRESHOLDS_FILE", str(MODELS / "photo-thresholds.json")))
THRESHOLD_VERSION = 3
PHOTO_THRESHOLD_VERSION = 1
PRODUCTION = os.environ.get("VERITY_PRODUCTION") == "1"
MAX_UPLOAD = 100 * 1024 * 1024
MAX_FRAMES = 8
MAX_VIDEO_SCAN_FRAMES = 240
MIME_TYPES = {
    "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
    "video/mp4": ".mp4", "video/webm": ".webm", "video/quicktime": ".mov",
}


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def pipeline_hashes() -> dict[str, str]:
    return {
        "pipeline_sha256": sha256(Path(__file__)),
        "clip_config_sha256": sha256(MODELS / "clip-config.json"),
        "clip_preprocessor_sha256": sha256(MODELS / "clip-preprocessor.json"),
    }


@lru_cache(maxsize=1)
def thresholds() -> tuple[float, float, bool]:
    if not THRESHOLDS_FILE.is_file():
        if PRODUCTION:
            raise ValueError("Validated thresholds are missing. Run the evaluation workflow in README.md.")
        return 0.3, 0.7, False
    config = json.loads(THRESHOLDS_FILE.read_text())
    if not isinstance(config, dict):
        raise ValueError("Threshold configuration is invalid.")
    try:
        real_max, fake_min = float(config["real_max"]), float(config["fake_min"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Threshold configuration is invalid.") from exc
    if config.get("version") != THRESHOLD_VERSION or not 0 <= real_max < fake_min <= 1:
        raise ValueError("Threshold configuration is invalid.")
    if config.get("model_sha256") != sha256(CHECKPOINT) or config.get("face_model_sha256") != sha256(FACE_MODEL):
        raise ValueError("Thresholds were validated with different model files.")
    if any(config.get(key) != value for key, value in pipeline_hashes().items()):
        raise ValueError("Thresholds were validated with a different inference pipeline.")
    return real_max, fake_min, True


@lru_cache(maxsize=1)
def photo_thresholds() -> tuple[float, float, bool]:
    if not PHOTO_THRESHOLDS_FILE.is_file():
        if PRODUCTION:
            raise ValueError("Validated photo thresholds are missing. Run the photo evaluation workflow in README.md.")
        return 0.05, 0.98, False
    config = json.loads(PHOTO_THRESHOLDS_FILE.read_text())
    try:
        real_max, fake_min = float(config["real_max"]), float(config["fake_min"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Photo threshold configuration is invalid.") from exc
    if (config.get("version") != PHOTO_THRESHOLD_VERSION or not 0 <= real_max < fake_min <= 1
            or config.get("model_sha256") != sha256(PHOTO_CHECKPOINT)
            or any(config.get(key) != value for key, value in photo_pipeline_hashes().items())):
        raise ValueError("Photo thresholds were validated with a different model or inference pipeline.")
    return real_max, fake_min, True


def classify(score: float | None, real_max: float, fake_min: float) -> str:
    if score is None:
        return "inconclusive"
    if score <= real_max:
        return "no_strong_signal"
    if score >= fake_min:
        return "likely_manipulated"
    return "inconclusive"


def read_video_frames(capture: cv2.VideoCapture) -> list[np.ndarray]:
    """Decode sequentially so codecs select the same frames on every platform."""
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total, MAX_VIDEO_SCAN_FRAMES) if total > 0 else MAX_VIDEO_SCAN_FRAMES
    wanted = set(np.linspace(0, limit - 1, min(MAX_FRAMES, limit), dtype=int))
    frames = []
    for index in range(limit):
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            if frame.shape[0] * frame.shape[1] > 20_000_000:
                raise ValueError("Video resolution exceeds 20 megapixels.")
            frames.append(frame)
    return frames


class GenDCLIP(nn.Module):
    """The released GenD CLIP module layout, loaded from its safetensors file."""

    def __init__(self) -> None:
        super().__init__()
        config = json.loads((MODELS / "clip-config.json").read_text())
        vision_config = CLIPVisionConfig(**config["vision_config"])
        self.feature_extractor = nn.Module()
        self.feature_extractor.vision_model = CLIPVisionModel(vision_config).vision_model
        self.feature_extractor.visual_projection = nn.Linear(
            vision_config.hidden_size, config["projection_dim"], bias=False
        )
        self.model = nn.Module()
        self.model.linear = nn.Linear(vision_config.hidden_size, 2)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor.vision_model(pixels).pooler_output
        return self.model.linear(F.normalize(features, p=2, dim=1))


class Detector:
    def __init__(self) -> None:
        if not CHECKPOINT.is_file():
            raise FileNotFoundError("GenD weights are missing. Run the download command in README.md.")
        selected = os.environ.get("VERITY_DEVICE") or (
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )
        if selected not in {"cpu", "mps", "cuda"}:
            raise ValueError("VERITY_DEVICE must be cpu, mps, or cuda.")
        self.device = torch.device(selected)
        self.model = GenDCLIP()
        load_model(self.model, CHECKPOINT, strict=True)
        self.model.to(self.device).eval()
        self.processor = CLIPImageProcessor.from_dict(
            json.loads((MODELS / "clip-preprocessor.json").read_text())
        )
        self.face_detector = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), 0.6, 0.3, 5000)

    def prepare_frame(self, frame: np.ndarray) -> tuple[Image.Image | None, int]:
        height, width = frame.shape[:2]
        scale = min(1.0, 960 / max(width, height))
        resized = cv2.resize(frame, (round(width * scale), round(height * scale))) if scale < 1 else frame
        self.face_detector.setInputSize((resized.shape[1], resized.shape[0]))
        _, faces = self.face_detector.detect(resized)
        if faces is None or len(faces) == 0:
            return None, 0

        # The most prominent face is analyzed when several people are present.
        face = max(faces, key=lambda row: row[2] * row[3])
        points = face[4:14].reshape(5, 2).astype(np.float32) / scale
        if points[0, 0] > points[1, 0]:
            points[[0, 1]] = points[[1, 0]]
        if points[3, 0] > points[4, 0]:
            points[[3, 4]] = points[[4, 3]]
        target = np.array(
            [[0.34, 0.46], [0.66, 0.46], [0.5, 0.64], [0.37, 0.82], [0.63, 0.82]],
            dtype=np.float32,
        )
        target = ((target - 0.5) / 1.3 + 0.5) * 256
        transform, _ = cv2.estimateAffinePartial2D(points, target, method=cv2.LMEDS)
        if transform is None:
            return None, len(faces)
        aligned = cv2.warpAffine(frame, transform, (256, 256))
        return Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)), len(faces)

    def score_frames(self, frames: list[np.ndarray]) -> list[tuple[float | None, int]]:
        prepared = [self.prepare_frame(frame) for frame in frames]
        images = [image for image, _ in prepared if image is not None]
        if not images:
            return [(None, count) for _, count in prepared]
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.inference_mode():
            scores = iter(self.model(pixels).softmax(dim=-1)[:, 1].tolist())
        return [(next(scores) if image is not None else None, count) for image, count in prepared]

    def score_frame(self, frame: np.ndarray) -> tuple[float | None, int]:
        return self.score_frames([frame])[0]

    def analyze(self, path: Path, is_video: bool) -> dict:
        if is_video:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise ValueError("This video could not be decoded.")
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width * height > 20_000_000:
                capture.release()
                raise ValueError("Video resolution exceeds 20 megapixels.")
            try:
                frames = read_video_frames(capture)
            finally:
                capture.release()
            if not frames:
                raise ValueError("No readable video frames were found.")
        else:
            try:
                with Image.open(path) as source:
                    if source.width * source.height > 20_000_000:
                        raise ValueError("Image resolution exceeds 20 megapixels.")
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    frames = [cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)]
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise ValueError("This image could not be decoded.") from exc

        frame_results = self.score_frames(frames)
        scores = [score for score, _ in frame_results if score is not None]
        multiple_faces = any(face_count > 1 for _, face_count in frame_results)
        fake_score = float(np.mean(scores)) if scores else None
        return {
            "fake_score": fake_score,
            "frames_sampled": len(frames),
            "frames_with_faces": len(scores),
            "multiple_faces": multiple_faces,
            "model": "GenD CLIP ViT-L/14",
        }


app = FastAPI(title="Verus Deepfake Detection API", docs_url="/api/docs", redoc_url=None)
app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")
_lock = threading.Lock()  # ponytail: serial inference; use a worker queue if concurrent traffic grows.
_detector: Detector | None = None
_photo_detector: PhotoDetector | None = None


@app.get("/")
def homepage():
    return FileResponse(FRONTEND / "index.html")


@app.get("/{filename}")
def frontend_file(filename: str):
    if filename not in {"styles.css", "script.js"}:
        raise HTTPException(404)
    return FileResponse(FRONTEND / filename)


@app.get("/api/health")
def health():
    global _detector, _photo_detector
    video_available = CHECKPOINT.is_file() and FACE_MODEL.is_file()
    photo_available = PHOTO_CHECKPOINT.is_file()
    model_available = video_available and photo_available
    video_calibrated = photo_calibrated = False
    try:
        _, _, video_calibrated = thresholds() if video_available else (0.3, 0.7, False)
        _, _, photo_calibrated = photo_thresholds() if photo_available else (0.05, 0.98, False)
        calibrated = video_calibrated and photo_calibrated
        problem = None
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        calibrated, problem = False, str(exc)
    ready = model_available and problem is None and (calibrated or not PRODUCTION)
    if ready and _detector is None:
        try:
            with _lock:
                if _detector is None:
                    _detector = Detector()
                if _photo_detector is None:
                    _photo_detector = PhotoDetector()
        except Exception:
            logging.exception("Detector initialization failed")
            ready, problem = False, "Detector could not load; check server logs."
    return {"model_available": model_available, "calibrated": calibrated,
            "video_calibrated": video_calibrated, "photo_calibrated": photo_calibrated,
            "ready": ready,
            "problem": problem}


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...)):
    global _detector, _photo_detector
    suffix = MIME_TYPES.get(file.content_type or "")
    if suffix is None:
        raise HTTPException(415, "Use a JPG, PNG, WebP, MP4, WebM, or MOV file.")
    is_video = suffix in {".mp4", ".webm", ".mov"}
    try:
        real_max, fake_min, calibrated = thresholds() if is_video else photo_thresholds()
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        raise HTTPException(503, str(exc)) from exc
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as target:
            path = Path(target.name)
            size = 0
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "The file must be under 100 MB.")
                target.write(chunk)
        if size == 0:
            raise HTTPException(422, "The file is empty.")
        with _lock:
            if is_video and _detector is None:
                try:
                    _detector = Detector()
                except FileNotFoundError as exc:
                    raise HTTPException(503, str(exc)) from exc
                except Exception as exc:
                    logging.exception("Detector initialization failed")
                    raise HTTPException(503, "Detector could not load; check server logs.") from exc
            if not is_video and _photo_detector is None:
                try:
                    _photo_detector = PhotoDetector()
                except FileNotFoundError as exc:
                    raise HTTPException(503, str(exc)) from exc
                except Exception as exc:
                    logging.exception("Photo detector initialization failed")
                    raise HTTPException(503, "Photo detector could not load; check server logs.") from exc
            try:
                result = _detector.analyze(path, True) if is_video else _photo_detector.analyze(path)
                result["verdict"] = classify(result["fake_score"], real_max, fake_min) if calibrated else "inconclusive"
                result["fake_score"] = round(result["fake_score"], 4) if result["fake_score"] is not None else None
                result["calibrated"] = calibrated
                result["media_type"] = "video" if is_video else "image"
                return result
            except (ValueError, cv2.error) as exc:
                raise HTTPException(422, str(exc)) from exc
    finally:
        file.file.close()
        if path is not None:
            path.unlink(missing_ok=True)
