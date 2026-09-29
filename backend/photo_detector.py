"""Detector for fully AI-generated still images."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import CLIPVisionConfig, CLIPVisionModel

MODELS = Path(__file__).resolve().parent / "models" / "ai-image"
CHECKPOINT = MODELS / "runC" / "checkpoints" / "best.pt"
CONFIG = MODELS / "clip-vit-b16-config.json"
SUMMARY = MODELS / "runC" / "summary.json"
MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(3, 1, 1)
STD = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(3, 1, 1)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def pipeline_hashes() -> dict[str, str]:
    return {"photo_pipeline_sha256": sha256(Path(__file__)), "photo_config_sha256": sha256(CONFIG)}


class AIImageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = CLIPVisionConfig(**json.loads(CONFIG.read_text())["vision_config"])
        self.backbone = CLIPVisionModel(config).vision_model
        self.head = nn.Sequential(nn.LayerNorm(config.hidden_size), nn.Dropout(0), nn.Linear(config.hidden_size, 1))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(pixel_values=pixels).pooler_output).flatten()


def canonical_image(image: Image.Image) -> Image.Image:
    """Apply the released model's documented training preprocessing."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = 320 / min(width, height)
    image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    width, height = image.size
    left, top = (width - 256) // 2, (height - 256) // 2
    image = image.crop((left, top, left + 256, top + 256))
    encoded = io.BytesIO()
    image.save(encoded, "JPEG", quality=95)
    encoded.seek(0)
    with Image.open(encoded) as decoded:
        return decoded.convert("RGB")


def five_crops(image: Image.Image) -> list[Image.Image]:
    image = canonical_image(image)
    return [image.crop(box) for box in ((0, 0, 224, 224), (32, 0, 256, 224),
                                        (0, 32, 224, 256), (32, 32, 256, 256),
                                        (16, 16, 240, 240))]


class PhotoDetector:
    def __init__(self) -> None:
        if not CHECKPOINT.is_file():
            raise FileNotFoundError("Photo detector weights are missing. Run the download command in README.md.")
        selected = os.environ.get("VERITY_DEVICE") or (
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )
        if selected not in {"cpu", "mps", "cuda"}:
            raise ValueError("VERITY_DEVICE must be cpu, mps, or cuda.")
        self.device = torch.device(selected)
        self.model = AIImageModel()
        blob = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        self.model.load_state_dict(blob["model"], strict=True)
        self.model.to(self.device).eval()
        self.temperature = float(json.loads(SUMMARY.read_text())["temperature"])
        if not self.temperature > 0:
            raise ValueError("Photo detector temperature must be positive.")

    def analyze(self, path: Path) -> dict:
        try:
            with Image.open(path) as source:
                if source.width * source.height > 20_000_000:
                    raise ValueError("Image resolution exceeds 20 megapixels.")
                crops = five_crops(source)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("This image could not be decoded.") from exc
        pixels = torch.stack([
            (torch.from_numpy(np.asarray(crop, dtype=np.float32).copy()).permute(2, 0, 1) / 255 - MEAN) / STD
            for crop in crops
        ]).to(self.device)
        with torch.inference_mode():
            logits = self.model(pixels).float().cpu() / self.temperature
        probabilities = logits.sigmoid()
        return {
            "fake_score": float(logits.mean().sigmoid()),
            "crop_spread": float(probabilities.std(correction=0)),
            "frames_sampled": 1,
            "frames_with_faces": 0,
            "multiple_faces": False,
            "model": "AI image CLIP ViT-B/16",
        }
